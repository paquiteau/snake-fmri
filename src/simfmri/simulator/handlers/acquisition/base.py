"""Base Acquisition handlers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from fmri.operators.fft import FFT

from simfmri.simulator.handlers.base import AbstractHandler
from simfmri.simulator.simulation import SimulationData
from simfmri.utils import validate_rng
from simfmri.utils.typing import RngType

from ._coils import get_smaps
from .trajectory import KspaceTrajectory, accelerate_TR


class AcquisitionHandler(AbstractHandler):
    r"""
    Simulate the acquisition of the data.

    Currently only Cartesian sampling is supported.

    Parameters
    ----------
    TR
        Time to acquire a full k-space.
    update_time
        sampling interval in the kpace. every update_time, the image is updated.
        Ideally, this should be the same as sim_tr.
    gen_smaps
        If true, smaps are also generated, default true.

    Notes
    -----
    The simulated data is sampled at the `SimulationData.sim_tr` interval, whereas the
    sampling mask is sampled at the update_time interval. This mean that the sampling
    mask can be more precise than the simulated data. In this case, the simulated data
    is interpolated to create a extra point. Every point in the mask that have the same
    sampling time will be handled together. Typically you want `update_time` to be equal
    to the time of the acquisition of one shot, and ideally equal to `sim_tr`.


    The MRI acquisition is modelled as follows:
    .. math::
        s(t) = \int_{\Omega} \rho(x,t) \exp(-2i\pi k(t) \cdot x) dx
    where :math:`\rho` is the object, :math:`k(t)` is the k-space trajectory.
    In practice:
    .. math::
        s(t) = \sum_{x \in \Omega} \rho(x,t) \exp(-2i\pi k(t) \cdot x) \Delta x
    """

    # TODO: add other trajectories sampling and refactor.


class VDSAcquisitionHandler(AcquisitionHandler):
    """
    Variable Density Sampling Acquisition Handler to generate k-space data.

    Parameters
    ----------
    acs
        Number/ proportion of lines to be acquired in the center of k-space.
    accel
        Acceleration factor for the rest of the lines.
    accel_axis
        Axis along which the acceleration is applied.
    direction
        Direction of the acquisition. Either "center-out" or "random".
    TR
        Time to acquire a full k-space. Exclusive with base_TR.
    base_TR
        Time to acquire a full k-space without acceleration. Exclusive with TR.
    pdf
        Probability density function of the sampling. Either "gaussian" or "uniform".
    rng
        Random number generator or seed.
    constant
        If true, the acceleration is constant along the axis. Otherwise, it is

    """

    def __init__(
        self,
        acs: float | int,
        accel: int,
        accel_axis: int,
        direction: Literal["center-out", "random"],
        TR: float = None,
        base_TR: float = None,
        pdf: Literal["gaussian", "uniform"] = "gaussian",
        rng: RngType = None,
        constant: bool = False,
        smaps: bool = True,
    ):
        super().__init__()
        rng = validate_rng(rng)
        self._traj_params = {
            "acs": acs,
            "accel": accel,
            "accel_axis": accel_axis,
            "direction": direction,
            "pdf": pdf,
            "rng": rng,
        }
        self.TR = accelerate_TR(TR, base_TR, accel)
        self.constant = constant
        self.smaps = smaps

    def _handle(self, sim: SimulationData) -> SimulationData:
        if self.TR > sim.sim_time:
            raise ValueError("TR should be smaller than sim_time.")
        if self.TR < sim.sim_tr:
            raise ValueError("TR should be larger than or equal to sim_tr.")
        upsampling = self.TR / sim.sim_tr

        if int(self.TR * 1000) % int(sim.sim_tr * 1000):
            self.log.warning("TR is not a multiple of sim_tr.")
        if self.smaps and sim.n_coils > 1:
            sim.smaps = get_smaps(sim.shape, sim.n_coils)

        # initialization of the frame variables.
        current_time = 0  # current time in the kspace
        current_time_frame = 0  # current time in the frame

        trajectory = KspaceTrajectory.vds(
            shape=sim.shape, TR=self.TR, **self._traj_params
        )
        kspace_data = []
        kspace_mask = []
        sim_frame = -1
        volume_kspace = np.squeeze(
            np.zeros((sim.n_coils, *sim.shape), dtype=np.complex64)
        )
        self.log.debug("trajectory has %s shots", len(trajectory._shots))
        self.log.debug("sim has %s frames", sim.n_frames)
        self.log.debug("expected number of frames %s", sim.sim_time / self.TR)
        self.log.debug(
            f"portion of kspace  updated at each sim frame: {sim.sim_tr / self.TR} "
            f"({trajectory.n_shots * sim.sim_tr/self.TR}/{trajectory.n_shots})"
        )

        if not np.isclose(int(trajectory.n_shots % upsampling), 0):
            self.log.warning(
                "Potential uneven repartition of shots"
                f"({trajectory.n_shots / upsampling})"
            )

        while current_time < sim.sim_time and sim_frame < sim.n_frames - 1:
            sim_frame += 1
            shot_selected = trajectory.extract_trajectory(
                current_time_frame, current_time_frame + sim.sim_tr
            )
            shots_mask = shot_selected.get_binary_mask(sim.shape)

            FFT_op = FFT(sim.shape, shots_mask, smaps=sim.smaps, n_coils=sim.n_coils)

            shots_kspace_data = FFT_op.op(sim.data_acq[sim_frame])
            if sim.n_coils > 1:
                pass
            else:
                volume_kspace += shots_kspace_data
            current_time += sim.sim_tr
            current_time_frame += sim.sim_tr
            if current_time_frame >= self.TR:
                # a full kspace has been acquired
                kspace_data.append(volume_kspace.copy())
                kspace_mask.append(trajectory.get_binary_mask(sim.shape))
                current_time_frame = 0
                volume_kspace = np.squeeze(
                    np.zeros((sim.n_coils, *sim.shape), dtype=np.complex64)
                )
                if not self.constant:
                    # new frame, new sampling
                    trajectory = KspaceTrajectory.vds(
                        shape=sim.shape,
                        TR=self.TR,
                        **self._traj_params,
                    )

        self.log.info(f"Acquired {len(kspace_data)} kspace volumes, at TR={self.TR} s.")
        sim.kspace_data = np.array(kspace_data)
        sim.kspace_mask = np.array(kspace_mask)
        sim.extra_infos["TR"] = self.TR
        sim.extra_infos["traj_name"] = "vds"
        sim.extra_infos["traj_params"] = self._traj_params
        return sim
