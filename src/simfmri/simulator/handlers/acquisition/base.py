"""Base Acquisition handlers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from fmri.operators.fft import FFT

from simfmri.utils import get_smaps

from ..base import AbstractHandler
from ..simulation import SimulationData
from ..utils import validate_rng
from ..utils.typing import RngType
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

    def _handle(self, sim: SimulationData) -> SimulationData:
        if self.TR > sim.sim_time:
            raise ValueError("TR should be smaller than sim_time.")
        if self.TR < sim.sim_tr:
            raise ValueError("TR should be larger than or equal to sim_tr.")

        if self.smaps:
            sim.smaps = get_smaps(sim.image, sim.n_coils)

        current_time = 0  # current time in the kspace
        current_time_frame = 0  # current time in the frame

        trajectory = KspaceTrajectory.vds(
            shape=sim.shape,
            TR=self.TR,
            **self._traj_params,
        )
        kspace_data = []
        sim_frame = -1
        while current_time < sim.sim_time:
            sim_frame += 1
            current_time_frame += sim.sim_tr
            shot_selected = trajectory.extract_trajectory(
                current_time_frame, current_time_frame + sim.sim_tr
            )
            volume_kspace = np.zeros((sim.n_coils, *sim.shape))
            shots_mask = shot_selected.get_binary_mask(
                sim.shape, current_time, current_time + sim.sim_tr
            )

            FFT_op = FFT(sim.shape, shots_mask, smaps=sim.smaps, n_coils=sim.n_coils)

            shots_kspace_data = FFT_op(sim.data_acq[sim_frame])
            volume_kspace[:, shots_mask] = shots_kspace_data

            current_time += sim.sim_tr
            if current_time_frame >= self.TR:
                # a full kspace has been acquired
                kspace_data.append(volume_kspace.copy())

                current_time_frame = 0
                if not self.constant:
                    # new frame, new sampling
                    trajectory = KspaceTrajectory.vds(
                        shape=sim.shape,
                        TR=self.TR,
                        **self._traj_params,
                    )

        sim.kspace_data = np.array(kspace_data)
        return sim
