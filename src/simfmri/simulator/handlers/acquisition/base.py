"""Base Acquisition handlers."""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

from hydra_callbacks import PerfLogger

from simfmri.simulator.handlers.base import AbstractHandler
from simfmri.simulator.simulation import SimulationData
from simfmri.utils import validate_rng
from simfmri.utils.typing import RngType
from fmri.operators.fourier import FFT_Sense
from mrinufft import get_operator

from ._coils import get_smaps
from .trajectory import KspaceTrajectory


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

    def __init__(self, constant: bool, smaps: bool, n_jobs: int):
        super().__init__()
        self.constant = constant
        self.smaps = smaps
        self.n_jobs = 4
        self.is_cartesian = True

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

        fft_op = FFT_Sense(data_sim.shape[1:], mask=mask, n_coils=n_coils, smaps=smaps)

        kspace_data[kspace_frame, ...] += fft_op.op(data_sim[sim_frame])
        return kspace_data, kspace_mask

    def _acquire(self, sim: SimulationData, trajectory_factory: Callable) -> np.ndarray:
        """Acquire the data by splitting the kspace shot over the simulation frames.

        This procedure is done in two steps:
        1. Plan the kspace trajectories
        - find the shots each simulation frame will consume
        - group the shots by kspace frame
        2. Execute the plan
        - Acquire each kspace frame in parallel.

        Parameters
        ----------
        sim
            The simulation data.
        trajectory_factory
            The factory to create the trajectory. This factory should return the
            trajectory for a single volume; and will be called for each

        """
        if self.smaps and sim.n_coils > 1:
            sim.smaps = get_smaps(sim.shape, sim.n_coils)

        plans, n_kspace_frame, TR_ms = self._plan(sim, trajectory_factory)
        kspace_data, kspace_mask = self._execute_plan(plans, n_kspace_frame, sim)

        self.log.info(f"Acquired {len(kspace_data)} kspace volumes, at TR={TR_ms}ms.")
        sim.kspace_data = np.array(kspace_data)
        sim.kspace_mask = np.array(kspace_mask)
        sim.extra_infos["TR_ms"] = TR_ms
        sim.extra_infos["traj_name"] = "vds"
        sim.extra_infos["traj_params"] = self._traj_params
        return sim

    def _execute_plan(
        self, plans: list[dict], n_kspace_frame: int, sim: SimulationData
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute the plan."""
        with PerfLogger(self.log, level=10, name="Execute Acquisition"):
            data_sim = sim.data_acq
            smaps = sim.smaps
            n_coils = sim.n_coils
            kspace_shape = (n_kspace_frame, n_coils, *sim.shape)

            kspace_data = np.squeeze(np.zeros(kspace_shape, dtype=np.complex64))
            kspace_mask = np.zeros((n_kspace_frame, *sim.shape), dtype=bool)
            for p in plans:
                kspace_data, kspace_mask = self.__execute_plan(
                    p, data_sim, kspace_data, kspace_mask, smaps, n_coils
                )
        return kspace_data, kspace_mask

    def _plan(
        self,
        sim: SimulationData,
        trajectory_factory: Callable,
    ) -> list[dict]:
        """Plan the acquisition."""
        trajectory = trajectory_factory(**self._traj_params)

        TR_ms = KspaceTrajectory.validate_TR(
            TR_ms=self._traj_params["TR_ms"],
            base_TR_ms=self._traj_params["base_TR_ms"],
            accel=self._traj_params["accel"],
            shot_time_ms=self._traj_params["shot_time_ms"],
            n_shot=trajectory.n_shots,
        )
        TR_ms = self._debug(sim, trajectory, TR_ms, self._traj_params["shot_time_ms"])
        n_kspace_frame = sim.sim_time_ms // TR_ms
        n_shot_per_sim_frame = int(trajectory.n_shots * sim.sim_tr_ms / TR_ms)

        plans = []
        shot_in_frame, shot_total = 0, 0
        sim_frame, kspace_frame = 0, 0
        with PerfLogger(self.log, level=10, name="Planning Acquisition"):  # 10 is DEBUG
            while shot_total < trajectory.n_shots * n_kspace_frame:
                if shot_in_frame + 2 * n_shot_per_sim_frame > trajectory.n_shots:
                    # the next run (after this one) will not have enough shots
                    # so we merge it with this one
                    # This is equivalent to duplicate the current sim_frame and
                    # use it for the remaining shots.
                    update = trajectory.n_shots - shot_in_frame
                else:
                    update = n_shot_per_sim_frame
                plans.append(
                    {
                        "shot_selected": trajectory.extract_trajectory(
                            shot_in_frame,
                            shot_in_frame + update,
                        ),
                        "sim_frame": sim_frame,
                        "kspace_frame": kspace_frame,
                    }
                )
                shot_in_frame += update
                shot_total += update
                sim_frame += 1
                if shot_in_frame >= trajectory.n_shots:
                    shot_in_frame = 0
                    kspace_frame += 1
                    if not self.constant:
                        # new frame, new sampling
                        trajectory = trajectory_factory(
                            **self._traj_params,
                        )
                assert shot_in_frame <= trajectory.n_shots
                assert sim_frame <= sim.n_frames
                assert kspace_frame <= n_kspace_frame

        self.log.debug("stopped at frame %s/%s", sim_frame, sim.n_frames)
        return plans, n_kspace_frame, TR_ms

    def _debug(
        self,
        sim: SimulationData,
        trajectory: KspaceTrajectory,
        TR_ms: int,
        shot_time_ms: int,
    ) -> None:
        """Print debug information about the trajectory."""
        if sim.sim_tr_ms % shot_time_ms != 0:
            self.log.warning(
                f"shot time {shot_time_ms}ms does not divide TR {sim.sim_tr_ms}ms."
            )
        if TR_ms % sim.sim_tr_ms != 0:
            old_TR_ms = TR_ms
            self.log.error(f"TR {sim.sim_tr_ms}ms does not divide shot time {TR_ms}ms.")
            TR_ms = sim.sim_tr_ms * (TR_ms // sim.sim_tr_ms)
            self.log.warning(
                f"Using TR={TR_ms}ms instead. (shot time {shot_time_ms * TR_ms/old_TR_ms }ms)"
            )
        self.log.debug(
            f"trajectory has {len(trajectory._shots)} shots, TR={TR_ms}ms\n"
            f"sim: {sim.n_frames} frames, @{sim.sim_tr_ms}ms, total {sim.sim_time_ms}\n"
            f"expected number of frames {sim.sim_time_ms // TR_ms}\n"
            f"portion of kspace updated at each sim frame:"
            f"{sim.sim_tr_ms / TR_ms}"
            f"({trajectory.n_shots * sim.sim_tr_ms/TR_ms}/{trajectory.n_shots})"
        )
        return TR_ms


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
        If true, the acceleration is constant along the axis.
        Otherwise, it is regenerated at each frame.
    smaps
        If true, apply sensitivity maps to the data.
    n_jobs
        Number of jobs to run in parallel for the frame acquisition.

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
        n_jobs: int = 4,
    ):
        super().__init__(constant, smaps, n_jobs)
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
        KspaceTrajectory.validate_TR(
            TR_ms,
            base_TR_ms,
            1,
            1,
            shot_time_ms,
        )

    def _handle(self, sim: SimulationData) -> SimulationData:
        self.traj_params["shape"] = sim.shape
        return self._acquire_variable(sim, trajectory_factory=KspaceTrajectory.vds)


class NonCartesianAcquisitionHandler(AcquisitionHandler):
    r"""
    Simulate the acquisition of the data.

    Non Cartesian sampling

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
    The simulated data is sampled at the `SimulationData.sim_tr` interval, whereas each
    shot is sampled at the update_time interval.
    Typically you want `update_time` to be equal or be a multiple of to the time of the
    acquisition of one shot, and ideally equal to `sim_tr`.


    The MRI acquisition is modelled as follows:
    .. math::
        s(t) = \int_{\Omega} \rho(x,t) \exp(-2i\pi k(t) \cdot x) dx
    where :math:`\rho` is the object, :math:`k(t)` is the k-space trajectory.
    In practice:
    .. math::
        s(t) = \sum_{x \in \Omega} \rho(x,t) \exp(-2i\pi k(t) \cdot x) \Delta x
    """

    # TODO: add other trajectories sampling and refactor.

    def __init__(self, constant: bool, smaps: bool, n_jobs: int):
        super().__init__()
        self.constant = constant
        self.smaps = smaps
        self.n_jobs = 4

    @staticmethod
    def __execute_plan(
        operatorKlass: Callable,
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
        fft_op = operatorKlass(
            samples=shot_selected,
            shape=data_sim.shape[1:],
            n_coils=n_coils,
            smaps=smaps,
        )

        kspace_data[kspace_frame, ...] += fft_op.op(data_sim[sim_frame])
        return kspace_data

    @staticmethod
    def __execute_plan(
        operator: str,
        plan: dict[str, Any],
        data_sim: np.ndarray,
        kspace_data: list,
        kspace_locs: list,
        smaps: np.ndarray,
        n_coils: int,
    ) -> None:
        shot_selected: KspaceTrajectory = plan["shot_selected"]
        sim_frame: int = plan["sim_frame"]
        kspace_frame: int = plan["kspace_frame"]
        kspace_locs[kspace_frame].append(shot_selected.shots)
        fourier_op = get_operator(operator)(
            shot_selected.shots, data_sim.shape[1:], n_coils=n_coils, smaps=smaps
        )

        kspace_data[kspace_frame].append(fourier_op.op(data_sim[sim_frame]))
        return kspace_data, kspace_locs

    def _execute_plan(
        self, plans: list[dict], n_kspace_frame: int, sim: SimulationData
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute the plan."""
        with PerfLogger(self.log, level=10, name="Execute Acquisition"):
            data_sim = sim.data_acq
            smaps = sim.smaps
            n_coils = sim.n_coils

            kspace_data = [[] for _ in range(n_kspace_frame)]
            kspace_locs = [[] for _ in range(n_kspace_frame)]
            for p in plans:
                kspace_data, kspace_locs = self.__execute_plan(
                    p, data_sim, kspace_data, kspace_locs, smaps, n_coils
                )
        kspace_data = np.array(kspace_data)
        kspace_locs = np.array(kspace_locs)
        return kspace_data, kspace_locs


class Radial2DAcquisitionHandler(NonCartesianAcquisitionHandler):
    """
    Radial 2D Acquisition Handler to generate k-space data.

    Parameters
    ----------
    Nc: int
        Number of shots in the radial pattern
    Ns: int
        Number of spokes in each shot
    smaps: bool
        If true, apply sensitivity maps to the data.
    angle: str
        If "constant", the trajectory is generated once and used for all frames.
        If "random", a random angle is generated for each frame.
    n_jobs: int
        Number of jobs to run in parallel for the frame acquisition.
    backend: str
        Backend to use for the Non Cartesian Fourier Transform default is "finufft"


    TODO: For Radial we could implement the Radon Transform.
    """

    def __init__(
        self,
        Nc: int,
        Ns: int,
        expansion: str = "rotation",
        nb_repeat: int = 1,
        smaps: bool = True,
        angle: str = "constant",
        n_jobs: int = 4,
    ) -> None:
        super().__init__(constant=angle == "constant", smaps=smaps, n_jobs=n_jobs)

        self._traj_params = {
            "Nc": Nc,
            "Ns": Ns,
            "expansion": expansion,
            "nb_repeat": nb_repeat,
        }

        self._angle = angle
        self._backend = "finufft"

    def _handle(self, sim: SimulationData) -> SimulationData:
        self.traj_params["dim"] = len(sim.shape)

        return self._acquire_variable(sim, trajectory_factory=KspaceTrajectory.radial)
