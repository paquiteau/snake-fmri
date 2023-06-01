"""Base Acquisition handlers."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Literal, Callable

import numpy as np
from fmri.operators.fft import FFT
from joblib import Parallel, delayed
from hydra_callbacks import PerfLogger


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

    @staticmethod
    def __execute_plan(
        plan: dict[str, Any],
        data_sim: np.ndarray,
        kspace_data: np.ndarray,
        kspace_mask: np.ndarray,
        smaps: np.ndarray,
        n_coils: int,
    ) -> None:
        shot_selected: KspaceTrajectory = plan["shot_selected"]
        sim_frame: int = plan["sim_frame"]
        kspace_frame: int = plan["kspace_frame"]
        mask = shot_selected.get_binary_mask(data_sim.shape[1:])
        kspace_mask[kspace_frame, ...] |= mask
        kspace_data[kspace_frame, ...] += FFT(
            data_sim.shape[1:],
            mask=mask,
            smaps=smaps,
            n_coils=n_coils,
        ).op(data_sim[sim_frame])

    def _acquire_variable(
        self, sim: SimulationData, trajectory_factory: Callable
    ) -> np.ndarray:
        """Acquire the data by splitting the kspace shot over the simulation frames.


        This procedure is done in two steps:
        1. Plan the kspace trajectories
        - find the shots each simulation frame will consume
        - group the shots by kspace frame
        2. Execute the plan
        - Acquire each kspace frame in parallel.

        """
        if self.smaps and sim.n_coils > 1:
            sim.smaps = get_smaps(sim.shape, sim.n_coils)

        # initialization of the frame variables.
        current_time = 0  # current time in the kspace in ms.
        current_time_frame = 0  # current time in the frame in ms.

        trajectory = trajectory_factory(shape=sim.shape, **self._traj_params)

        sim_frame = -1
        kspace_frame = 0
        TR_ms = KspaceTrajectory.validate_TR(
            TR_ms=self._traj_params["TR_ms"],
            base_TR_ms=self._traj_params["base_TR_ms"],
            accel=self._traj_params["accel"],
            shot_time_ms=self._traj_params["shot_time_ms"],
            n_shot=trajectory.n_shots,
        )
        self._debug(sim, trajectory, TR_ms)
        plans = []
        # 1. Plan the kspace trajectories
        with PerfLogger(self.log, level=10, name="Planning"):  # 10 is DEBUG
            while current_time < sim.sim_time_ms and sim_frame < sim.n_frames - 1:
                sim_frame += 1
                shot_selected = trajectory.extract_trajectory(
                    current_time_frame, current_time_frame + sim.sim_tr
                )
                plans.append(
                    {
                        "shot_selected": shot_selected,
                        "sim_frame": sim_frame,
                        "kspace_frame": kspace_frame,
                    }
                )
                current_time += sim.sim_tr_ms
                current_time_frame += sim.sim_tr_ms
                if current_time_frame >= TR_ms:
                    # a full kspace has been acquired
                    kspace_frame += 1
                    current_time_frame = 0
                    if not self.constant:
                        # new frame, new sampling
                        trajectory = trajectory_factory(
                            shape=sim.shape,
                            **self._traj_params,
                        )
        # 2. Execute the plans using joblib
        with PerfLogger(self.log, level=10, name="Execution"):
            data_sim = sim.data_acq
            smaps = sim.smaps
            n_coils = sim.n_coils

            path = Path("/tmp/vdsjoblib/")
            path.mkdir(parents=True, exist_ok=True)

            kspace_data = np.squeeze(
                np.memmap(
                    filename=path / "kspace_data",
                    shape=(int(sim.sim_time_ms / TR_ms), n_coils, *sim.shape),
                    dtype=np.complex64,
                    mode="w+",
                )
            )

            kspace_mask = np.squeeze(
                np.memmap(
                    filename=path / "kspace_mask",
                    shape=(int(sim.sim_time_ms / TR_ms), n_coils, *sim.shape),
                    dtype=np.bool,
                    mode="w+",
                )
            )

            self.log.debug("Executing plans")
            Parallel(n_jobs=-1, verbose=0)(
                delayed(self.__execute_plan)(
                    plan, data_sim, kspace_data, kspace_mask, smaps, n_coils
                )
                for plan in plans
            )
            try:
                shutil.rmtree(path)
            except:  # noqa
                self.log.warning("Could not delete temporary folder")

        self.log.info(f"Acquired {len(kspace_data)} kspace volumes, at TR={TR_ms} s.")
        sim.kspace_data = np.array(kspace_data)
        sim.kspace_mask = np.array(kspace_mask)
        sim.extra_infos["TR"] = TR_ms
        sim.extra_infos["traj_name"] = "vds"
        sim.extra_infos["traj_params"] = self._traj_params
        return sim

    def _debug(
        self, sim: SimulationData, trajectory: KspaceTrajectory, TR_ms: int
    ) -> None:
        if TR_ms > sim.sim_time_ms:
            raise ValueError("TR should be smaller than sim_time.")
        if TR_ms < sim.sim_tr:
            raise ValueError("TR should be larger than or equal to sim_tr.")
        upsampling = TR_ms / sim.sim_tr_ms

        if TR_ms % sim.sim_tr_ms:
            self.log.warning("TR is not a multiple of sim_tr.")

        self.log.debug("trajectory has %s shots", len(trajectory._shots))
        self.log.debug("sim has %s frames", sim.n_frames)
        self.log.debug("expected number of frames %s", sim.sim_time_ms / TR_ms)
        self.log.debug(
            f"portion of kspace  updated at each sim frame: {sim.sim_tr_ms / TR_ms} "
            f"({trajectory.n_shots * sim.sim_tr_ms/TR_ms}/{trajectory.n_shots})"
        )

        if not np.isclose(int(trajectory.n_shots % upsampling), 0):
            self.log.warning(
                "Potential uneven repartition of shots"
                f"({trajectory.n_shots / upsampling})"
            )


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
        TR_ms: int = None,
        base_TR_ms: int = None,
        shot_time_ms: int = None,
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
            "TR_ms": TR_ms,
            "base_TR_ms": base_TR_ms,
            "shot_time_ms": shot_time_ms,
        }
        self.constant = constant
        self.smaps = smaps

        KspaceTrajectory.validate_TR(
            TR_ms,
            base_TR_ms,
            1,
            1,
            shot_time_ms,
        )

    def _handle(self, sim: SimulationData) -> SimulationData:
        return self._acquire_variable(sim, trajectory_factory=KspaceTrajectory.vds)
