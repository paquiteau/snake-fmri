"""Base Acquisition handlers."""

from __future__ import annotations

from typing import Any, Literal, Mapping
from collections.abc import Generator

import numpy as np

from hydra_callbacks import PerfLogger

from simfmri.simulator.handlers.base import AbstractHandler
from simfmri.simulator.simulation import SimDataType
from simfmri.utils import validate_rng
from simfmri.utils.typing import RngType
from fmri.operators.fourier import FFT_Sense
from mrinufft import get_operator

from .trajectory import (
    vds_factory,
    radial_factory,
    TrajectoryGeneratorType,
    trajectory_generator,
    rotate_trajectory,
    kspace_bulk_shot,
)

from ._coils import get_smaps
from .workers import acquire_cartesian_mp


SimGeneratorType = Generator[np.ndarray, None, None]


def _get_slicer(shot: np.ndarray) -> tuple[slice, slice, slice]:
    """Return a slicer for the mask.

    Fully sampled axis are marked with a -1.
    """
    slicer = [slice(None, None, None)] * shot.shape[-1]
    accel_axis = [i for i, v in enumerate(shot[0]) if v != -1][0]
    slicer[accel_axis] = shot[0][accel_axis]
    return tuple(slicer)


def _acquire_single_cartesian(
    sim_frame: np.ndarray,
    shot_batch: np.ndarray,
    shot_in_kspace_frame: np.ndarray,
    smaps: np.ndarray | None,
    n_coils: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = np.zeros((len(shot_batch), *sim_frame.shape), dtype=np.int8)
    for i, shot in enumerate(shot_batch):
        masks[i][_get_slicer(shot)] = 1
    mask = np.sum(masks, axis=0)
    fourier_op = FFT_Sense(sim_frame.shape, mask=mask, smaps=smaps, n_coils=n_coils)
    return fourier_op.op(sim_frame), masks, shot_in_kspace_frame


def _gen_kspace_cartesian(
    sim: SimDataType,
    shots_generator: TrajectoryGeneratorType,
    **kwargs: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for sim_frame_idx, shot_batch, shot_in_kframe in shots_generator:
        yield _acquire_single_cartesian(
            np.complex64(sim.data_acq[sim_frame_idx]),
            shot_batch,
            shot_in_kframe,
            smaps=sim.smaps,
            n_coils=sim.n_coils,
        )


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
        self.n_jobs = n_jobs
        self.is_cartesian = True

    def _acquire_cartesian(
        self,
        sim: SimDataType,
        trajectory_generator: TrajectoryGeneratorType,
        n_shot_sim_frames: int,
        n_kspace_frames: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Acquire cartesiant data and mask. Using multi processing."""
        kspace_data = np.zeros(
            (n_kspace_frames, sim.n_coils, *sim.shape), dtype=np.complex64
        )
        kspace_mask = np.zeros((n_kspace_frames, *sim.shape), dtype=np.int8)

        shot_gen = kspace_bulk_shot(trajectory_generator, n_shot_sim_frames)

        kspace_cartesian = _gen_kspace_cartesian(
            sim,
            shot_gen,
        )

        for kspace_val, kspace_masks, shot_in_kspace_frame in kspace_cartesian:
            for mask, ks_frame in zip(kspace_masks, shot_in_kspace_frame):
                kspace_data[ks_frame, :] += mask * kspace_val
                kspace_mask[ks_frame] |= mask

        return kspace_data, kspace_mask
        # trajectory_generator returns (shots, kspace_frame)
        #
        # Parallel pool of worker / Queue
        #   Get  sim.data_acq[sim_frame]
        #

    def _acquire(
        self,
        sim: SimDataType,
        trajectory_generator: TrajectoryGeneratorType,
    ) -> np.ndarray:
        """Acquire the data by splitting the kspace shot over the simulation frames.

        Parameters
        ----------
        sim
            The simulation data.
        trajectory_factory
            The factory to create the trajectory. This factory should return the
            trajectory for a single volume. and takes **self.traj_params as input.
        """
        if self.smaps and sim.n_coils > 1:
            sim.smaps = get_smaps(sim.shape, sim.n_coils).astype(np.complex64)

        test_traj = next(trajectory_generator)
        frame_TR_ms = self._validate_TR(
            sim, len(test_traj), self._traj_params["shot_time_ms"]
        )
        if sim.sim_time_ms % frame_TR_ms:
            self.log.warning("the TR for a frame does not divide the simulation time.")

        n_kspace_frame = int(sim.sim_time_ms / frame_TR_ms)
        n_shot_sim_frame = (n_kspace_frame * len(test_traj)) // sim.n_frames
        self.log.debug("n_shot/sim_frame %d/%d", n_shot_sim_frame, sim.n_frames)

        sim.extra_infos["TR_ms"] = frame_TR_ms
        sim.extra_infos["traj_name"] = "vds"
        sim.extra_infos["traj_constant"] = self.constant
        sim.extra_infos["traj_params"] = self._traj_params

        kspace_data, kspace_mask = acquire_cartesian_mp(
            sim, trajectory_generator, n_shot_sim_frame, n_kspace_frame, self.n_jobs
        )

        sim.kspace_data = kspace_data
        sim.kspace_mask = kspace_mask
        return sim

    def _validate_TR(
        self,
        sim: SimDataType,
        n_shots: int,
        shot_time_ms: int,
    ) -> None:
        """Print debug information about the trajectory."""
        TR_ms = shot_time_ms * n_shots
        if sim.sim_tr_ms % shot_time_ms != 0:
            self.log.warning(
                f"shot time {shot_time_ms}ms does not divide TR {sim.sim_tr_ms}ms."
            )
        if TR_ms % sim.sim_tr_ms != 0:
            old_TR_ms = TR_ms
            self.log.error(
                f"simTR {sim.sim_tr_ms}ms does not divide total shot time {TR_ms}ms."
            )
            TR_ms = sim.sim_tr_ms * (TR_ms // sim.sim_tr_ms)
            shot_time_ms = shot_time_ms * TR_ms / old_TR_ms
            self.log.warning(
                f"Using TR={TR_ms}ms instead. (shot time {shot_time_ms}ms)"
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
            "shot_time_ms": shot_time_ms,
        }

    def _handle(self, sim: SimDataType) -> SimDataType:
        self._traj_params["shape"] = sim.shape
        return self._acquire(
            sim,
            trajectory_generator=trajectory_generator(vds_factory, **self._traj_params),
        )


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
    def __init__(
        self,
        constant: bool = False,
        smaps: bool = True,
        n_jobs: int = 4,
        backend: str = "finufft",
    ):
        super().__init__(constant, smaps, n_jobs)
        self._backend = backend

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
        shot_selected = plan["shot_selected"]
        sim_frame: int = plan["sim_frame"]
        kspace_frame: int = plan["kspace_frame"]
        kspace_locs[kspace_frame].append(shot_selected.shots)
        fourier_op = get_operator(operator)(
            shot_selected.shots, data_sim.shape[1:], n_coils=n_coils, smaps=smaps
        )

        kspace_data[kspace_frame].append(fourier_op.op(data_sim[sim_frame]).copy())
        del fourier_op
        return kspace_data, kspace_locs

    def _execute_plan(
        self, plans: list[dict], n_kspace_frame: int, sim: SimDataType
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute the plan."""
        with PerfLogger(self.log, level=10, name="Execute Acquisition"):
            data_sim = sim.data_acq.astype(np.complex64)
            smaps = sim.smaps
            n_coils = sim.n_coils

            kspace_data = [[] for _ in range(n_kspace_frame)]
            kspace_locs = [[] for _ in range(n_kspace_frame)]
            for p in plans:
                kspace_data, kspace_locs = self.__execute_plan(
                    self._backend, p, data_sim, kspace_data, kspace_locs, smaps, n_coils
                )

        kspace_data = np.array(kspace_data)
        if sim.n_coils > 1:
            kspace_data = np.moveaxis(kspace_data, (0, 1, 2, 3), (0, 2, 1, 3))
            kspace_data = kspace_data.reshape(n_kspace_frame, sim.n_coils, -1)
        else:
            kspace_data = kspace_data.reshape(n_kspace_frame, -1)
        kspace_data = np.ascontiguousarray(kspace_data)

        kspace_locs = np.array(kspace_locs)
        kspace_locs = kspace_locs.reshape(n_kspace_frame, -1, *kspace_locs.shape[-2:])
        kspace_locs = np.ascontiguousarray(kspace_locs)

        return kspace_data, kspace_locs


class RadialAcquisitionHandler(NonCartesianAcquisitionHandler):
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
        n_shots: int,
        n_points: int,
        expansion: str = "rotation",
        n_repeat: int = 1,
        smaps: bool = True,
        angle: str = "constant",
        n_jobs: int = 4,
        shot_time_ms: int = 20,
        backend: str = "finufft",
    ) -> None:
        super().__init__(
            constant=angle == "constant", smaps=smaps, n_jobs=n_jobs, backend=backend
        )

        self._traj_params = {
            "n_shots": n_shots,
            "n_points": n_points,
            "expansion": expansion,
            "n_repeat": n_repeat,
            "shot_time_ms": shot_time_ms,
            #            "TR_ms": shot_time_ms * n_shots,
        }

        self._angle = angle

    def _handle(self, sim: SimDataType) -> SimDataType:
        self._traj_params["dim"] = len(sim.shape)
        self._traj_params[""]
        sim.extra_infos["operator"] = self._backend

        return self._acquire(
            sim,
            trajectory_generator=rotate_trajectory(
                trajectory_generator(radial_factory, **self._traj_params),
                self._angle,
            ),
        )
