"""Base Acquisition handlers."""

from __future__ import annotations

import logging
from collections.abc import Generator
from itertools import cycle
from typing import Any, Literal, Callable

import numpy as np
from mrinufft.io import read_trajectory

from snkf.simulation import SimData
from snkf.utils import AnyShape, validate_rng

from ..base import AbstractHandler, requires_field
from ._coils import get_smaps
from ._tools import TrajectoryFactoryProtocol, TrajectoryGeneratorType
from .trajectory import (
    radial_factory,
    rotate_trajectory,
    stack_spiral_factory,
    vds_factory,
)
from .workers import acq_cartesian, acq_noncartesian

logger = logging.getLogger("simulation.acquisition")


requires_data_acq = requires_field("data_acq", lambda sim: sim.data_ref.copy())


@requires_data_acq
class BaseAcquisitionHandler(AbstractHandler):
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
    The simulated data is sampled at the `SimData.sim_tr` interval, whereas the
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

    acquire_mp = staticmethod(acq_cartesian)
    cartesian = True

    def __init__(self, constant: bool, smaps: bool, shot_time_ms: int):
        super().__init__()
        self.constant = constant
        self.smaps = smaps
        self.shot_time_ms = shot_time_ms
        self._traj_params: dict[str, Any] = dict()

    def _acquire(
        self,
        sim: SimData,
        trajectory_generator: TrajectoryGeneratorType,
        **kwargs: Any,
    ) -> SimData:
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
            self.log.debug(f"generating sensitivity maps {sim.shape}, {sim.n_coils}")
            sim.smaps = get_smaps(sim.shape, sim.n_coils).astype(np.complex64)

        test_traj = next(trajectory_generator)
        n_shot_traj = len(test_traj)

        self.log.info(f"full trajectory has {n_shot_traj} shots")
        if sim.sim_tr_ms % self.shot_time_ms != 0:
            # find the closest shot time that divides the simulation time
            new_shot_time_ms = int(
                sim.sim_tr_ms / ((sim.sim_tr_ms // self.shot_time_ms) + 1)
            )
            self.log.warning(
                f"shot time {self.shot_time_ms} (ms) does not divide"
                f"sim tr {sim.sim_tr_ms} (ms)."
                f" Updating to {new_shot_time_ms} (ms) per shot."
            )
            self.shot_time_ms = new_shot_time_ms

        kframe_tr = n_shot_traj * self.shot_time_ms
        self.log.debug("initial TR volume %i ms", kframe_tr)

        n_shot_sim_frame = int(sim.sim_tr_ms / self.shot_time_ms)
        if n_shot_sim_frame > n_shot_traj:
            raise ValueError(
                "Not enough shots in trajectory to allow multiple simframe per kspace."
                "Increase number of shot per trajectory or use shorter time resolution."
            )
        n_tot_shots = sim.n_frames * n_shot_sim_frame
        n_kspace_frame = int(np.ceil(n_tot_shots / n_shot_traj))

        self.log.debug("n_kspace_frame %d", n_kspace_frame)
        self.log.debug("n_tot_shots %d", n_tot_shots)
        self.log.debug("n_shot_traj %d", n_shot_traj)
        self.log.debug("n_shot_sim_frame %d", n_shot_sim_frame)

        sim.extra_infos["TR_ms"] = kframe_tr
        sim.extra_infos["traj_name"] = "vds"
        sim.extra_infos["traj_constant"] = self.constant
        sim.extra_infos["traj_params"] = self._traj_params
        sim.extra_infos["n_shot_per_frame"] = n_shot_traj

        kspace_data, kspace_mask = self.acquire_mp(
            sim,
            trajectory_generator,
            n_shot_sim_frame,
            n_kspace_frame,
            **kwargs,
        )

        if np.ceil(n_tot_shots / n_shot_traj) >= n_tot_shots / n_shot_traj:
            # the last frame is not full and should be removed
            kspace_data = kspace_data[:-1]
            kspace_mask = kspace_mask[:-1]

        sim.kspace_data = kspace_data
        sim.kspace_mask = kspace_mask
        return sim


def trajectory_generator(
    traj_factory: Callable[..., np.ndarray],
    shape: AnyShape,
    **kwargs: Any,
) -> Generator[np.ndarray, None, None]:
    """Generate a trajectory.

    Parameters
    ----------
    traj_factory
        Trajectory factory function.
    n_batch
        Number of shot to deliver at once.
    kwargs
        Trajectory factory kwargs.

    Yields
    ------
    np.ndarray
        Kspace trajectory.
    """
    if kwargs.pop("constant", False):
        logger.debug("constant trajectory")
        traj = traj_factory(shape, **kwargs)
        while True:
            yield traj
    while True:
        yield traj_factory(shape, **kwargs)


class VDSAcquisitionHandler(BaseAcquisitionHandler):
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

    """

    name = "acquisition-vds"

    def __init__(
        self,
        acs: float | int,
        accel: int,
        accel_axis: int,
        direction: Literal["center-out", "random"],
        shot_time_ms: int,
        pdf: Literal["gaussian", "uniform"] = "gaussian",
        constant: bool = False,
        smaps: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(constant=constant, shot_time_ms=shot_time_ms, smaps=smaps)
        self._traj_params = {
            "acs": acs,
            "accel": accel,
            "accel_axis": accel_axis,
            "direction": direction,
            "pdf": pdf,
        }

        self.n_jobs = n_jobs

    def _handle(self, sim: SimData) -> SimData:
        self._traj_params["rng"] = validate_rng(sim.rng)
        return self._acquire(
            sim,
            trajectory_generator=trajectory_generator(
                vds_factory, sim.shape, **self._traj_params
            ),
            n_jobs=self.n_jobs,
        )


class NonCartesianAcquisitionHandler(BaseAcquisitionHandler):
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
    The simulated data is sampled at the `SimData.sim_tr` interval, whereas each
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

    acquire_mp = staticmethod(acq_noncartesian)

    # TODO: add other trajectories sampling and refactor.
    def __init__(
        self,
        constant: bool = False,
        smaps: bool = True,
        backend: str = "finufft",
        shot_time_ms: int = 50,
        n_jobs: int = 4,
    ):
        super().__init__(constant=constant, smaps=smaps, shot_time_ms=shot_time_ms)
        self._backend = backend
        self.n_jobs = n_jobs


class GenericAcquisitionHandler(BaseAcquisitionHandler):
    """
    Generic Acquisition Handler to generate k-space data.

    Parameters
    ----------
    traj_factory: TrajectoryFactoryProtocol
        The factory to create the trajectory. This factory should return the trajectory
        for a single volume. The first argument is the volume shape of the simuluation,
        the other arguments are given as kwargs from ``traj_params``.
    traj_params: Mapping[str, Any]
        The parameters to pass to the trajectory factory.
    shot_time_ms: int
        Time to acquire a single shot in ms.
    traj_generator: TrajectoryGeneratorType
    constant: bool, default True
        If true, the trajectory is generated once and used for all frames.
        Otherwise, it is regenerated at each frame.
    smaps: bool, default True
        If true, apply sensitivity maps to the data.

    Notes
    -----
    The trajectory factory is responsible to generate the trajectory for a single kspace
    frame. This factory is called by the trajectory generator, which order generation of
    trajectory, and applies transformation if wanted.


    """

    name = "acquisition-generic"

    def __init__(
        self,
        trajectory_factory: TrajectoryFactoryProtocol,
        traj_params: dict[str, Any],
        shot_time_ms: int,
        traj_generator: TrajectoryGeneratorType | None = None,
        cartesian: bool = True,
        smaps: bool = True,
        constant: bool = True,
        n_jobs: int = 1,
    ):
        self.acquire_mp = staticmethod(
            acq_cartesian if self.cartesian else acq_noncartesian
        )
        self.traj_factory = trajectory_factory
        self.traj_generator = traj_generator or trajectory_generator
        self.shot_time_ms = shot_time_ms
        self._traj_params = traj_params
        self.n_jobs = n_jobs

    def _handle(self, sim: SimData) -> SimData:
        return self._acquire(
            sim,
            trajectory_generator=self.traj_generator(
                self.traj_factory, sim.shape, **self._traj_params
            ),
            n_jobs=self.n_jobs,
        )


class GenericNonCartesianAcquisitionHandler(NonCartesianAcquisitionHandler):
    """Generic Acquisition handlers based on a list of files."""

    name = "acquisition-generic-noncartesian"

    def __init__(self, traj_files: list[str] | str, smaps: bool, shot_time_ms: int):
        if isinstance(traj_files, str):
            traj_files = [traj_files]

        super().__init__(
            constant=len(traj_files) == 1, shot_time_ms=shot_time_ms, smaps=smaps
        )
        self.traj_files = traj_files

    def _handle(self, sim: SimData) -> SimData:
        return self._acquire(sim, trajectory_generator=self.traj_generator)

    def traj_generator(self) -> Generator[np.ndarray, None, None]:
        """Generate trajectory by cycling over the files."""
        for file in cycle(self.traj_files):
            traj, params = read_trajectory(file)
            yield traj


class RadialAcquisitionHandler(NonCartesianAcquisitionHandler):
    """
    Radial 2D Acquisition Handler to generate k-space data.

    Parameters
    ----------
    Nc: int
        Number of shots in the radial pattern
    Ns: int
        Number of spokes in each shot
        If true, apply sensitivity maps to the data.
    angle: str
        If "constant", the trajectory is generated once and used for all frames.
        If "random", a random angle is generated for each frame
    **kwargs:
        Extra arguments (smaps, n_jobs, backend etc...)

    TODO: For Radial we could implement the Radon Transform ?
    """

    name = "acquisition-radial"

    def __init__(
        self,
        n_shots: int,
        n_points: int,
        expansion: str = "rotation",
        n_repeat: int = 1,
        angle: str = "constant",
        shot_time_ms: int = 20,
        smaps: bool = True,
        backend: str = "finufft",
        n_jobs: int = 4,
    ) -> None:
        super().__init__(
            constant=angle == "constant",
            shot_time_ms=shot_time_ms,
            smaps=smaps,
            backend=backend,
            n_jobs=n_jobs,
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

    def _handle(self, sim: SimData) -> SimData:
        sim.extra_infos["operator"] = self._backend

        return self._acquire(
            sim,
            trajectory_generator=rotate_trajectory(
                trajectory_generator(radial_factory, sim.shape, **self._traj_params),
                self._angle,
            ),
            n_jobs=self.n_jobs,
        )


class StackedSpiralAcquisitionHandler(NonCartesianAcquisitionHandler):
    """
    Spiral 2D Acquisition Handler to generate k-space data.

    Parameters
    ----------
    acsz: float | int
        Number/ proportion of lines to be acquired in the center of k-space.
    accelz: int
        Acceleration factor for the rest of the lines.
    directionz: Literal["center-out", "random"]
        Direction of the acquisition. Either "center-out" or "random".
    pdfz: Literal["gaussian", "uniform"]
        Probability density function of the sampling. Either "gaussian" or "uniform".
    shot_time_ms: int
        Time to acquire a single spiral in plane.
    n_samples: int
        Number of samples in the spiral.
    nb_revolutions: int
        Number of revolutions of the spiral.
    in_out: bool
        If true, the spiral is acquired with a double join pattern from/to the periphery
    **kwargs:
        Extra arguments (smaps, n_jobs, backend etc...)
    """

    name = "acquisition-sos"

    def __init__(
        self,
        acsz: float | int,
        accelz: int,
        directionz: Literal["center-out", "random"],
        n_samples: int = 3000,
        nb_revolutions: int = 10,
        in_out: bool = True,
        pdfz: Literal["gaussian", "uniform"] = "gaussian",
        constant: bool = False,
        smaps: bool = True,
        backend: str = "finufft",
        shot_time_ms: int = 50,
        n_jobs: int = 4,
    ):
        super().__init__(
            constant=constant,
            backend=backend,
            shot_time_ms=shot_time_ms,
            n_jobs=n_jobs,
        )
        self._traj_params = {
            "acsz": acsz,
            "accelz": accelz,
            "directionz": directionz,
            "pdfz": pdfz,
            "n_samples": n_samples,
            "nb_revolutions": nb_revolutions,
            "in_out": in_out,
        }

    def _handle(self, sim: SimData) -> SimData:
        self._traj_params["shape"] = sim.shape
        self._traj_params["rng"] = validate_rng(sim.rng)

        sim.extra_infos["operator"] = "stacked-" + self._backend

        return self._acquire(
            sim,
            trajectory_generator=trajectory_generator(
                stack_spiral_factory, **self._traj_params
            ),
            # extra kwargs for the nufft operator
            op_backend="stacked",
            z_index="auto",
            backend=self._backend,
            n_jobs=self.n_jobs,
        )
