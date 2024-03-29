"""Base Acquisition handlers."""

from __future__ import annotations

import logging
import dataclasses
from collections.abc import Generator
from itertools import cycle
from typing import Any, Callable

import numpy as np
from mrinufft.io import read_trajectory

from snkf.simulation import SimData
from snkf.base import AnyShape, validate_rng

from ..base import AbstractHandler, requires_field
from ._coils import get_smaps
from ._tools import TrajectoryGeneratorType
from .trajectory import (
    radial_factory,
    rotate_trajectory,
    stack_spiral_factory,
    vds_factory,
    AngleRotation,
)
from .cartesian_sampling import VDSorder, VDSpdf
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

    shot_time_ms: int
    constant: bool = False
    smaps: bool = True
    n_jobs: int = 1
    mock = False
    acquire_mp = staticmethod(acq_cartesian)

    def __post_init__(self):
        super().__post_init__()
        self._traj_params = {"constant": self.constant}

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
        sim.extra_infos["traj_params"] = self._traj_params
        sim.extra_infos["n_shot_per_frame"] = n_shot_traj

        # early stopping
        if self.mock:
            return sim

        if self.smaps and sim.n_coils > 1:
            self.log.debug(f"generating sensitivity maps {sim.shape}, {sim.n_coils}")
            sim.smaps = get_smaps(sim.shape, sim.n_coils).astype(np.complex64)

        kspace_data, kspace_mask = self.acquire_mp(
            sim,
            trajectory_generator,
            n_shot_sim_frame,
            n_kspace_frame,
            n_jobs=self.n_jobs,
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

    __handler_name__ = "acquisition-vds"

    acs: float | int
    accel: int
    accel_axis: int
    shot_time_ms: int
    order: VDSorder = VDSorder.CENTER_OUT
    pdf: VDSpdf = VDSpdf.GAUSSIAN
    constant: bool = False
    smaps: bool = True
    n_jobs: int = 1

    def __post_init__(
        self,
    ):
        super().__post_init__()
        self._traj_params |= {
            "acs": self.acs,
            "accel": self.accel,
            "accel_axis": self.accel_axis,
            "order": self.order,
            "pdf": self.pdf,
        }

    def _handle(self, sim: SimData) -> SimData:
        self._traj_params["rng"] = validate_rng(sim.rng)
        return self._acquire(
            sim,
            trajectory_generator=trajectory_generator(
                vds_factory, sim.shape, **self._traj_params
            ),
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

    constant: bool = False
    smaps: bool = True
    backend: str = "finufft"
    shot_time_ms: int = 50
    n_jobs: int = 4


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

    __handler_name__ = "acquisition-generic"

    traj_factory: str
    traj_params: dict[str, Any]
    shot_time_ms: int
    traj_generator: str
    cartesian: bool = True
    smaps: bool = True
    constant: bool = True
    n_jobs: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.acquire_mp = staticmethod(
            acq_cartesian if self.cartesian else acq_noncartesian
        )
        self._traj_params |= self.traj_params
        self.traj_factory = eval(self.traj_factory)
        self.traj_generator = eval(self.traj_generator)

    def _handle(self, sim: SimData) -> SimData:
        return self._acquire(
            sim,
            trajectory_generator=self.traj_generator(
                self.traj_factory, sim.shape, **self._traj_params
            ),
        )


class GenericNonCartesianAcquisitionHandler(NonCartesianAcquisitionHandler):
    """Generic Acquisition handlers based on a list of files."""

    __handler_name__ = "acquisition-generic-noncartesian"

    traj_files: list[str]
    smaps: bool
    shot_time_ms: int
    backend: str
    traj_osf: int = 1
    n_jobs: int = 1

    def __post_init__(self):
        if isinstance(self.traj_files, str):
            self.traj_files = [self.traj_files]
        if len(self.traj_files) == 1:
            self.constant = True
        super().__post_init__()

    def _handle(self, sim: SimData) -> SimData:
        return self._acquire(
            sim,
            trajectory_generator=self.traj_generator(),
            backend_name=self.backend,
        )

    def traj_generator(self) -> Generator[np.ndarray, None, None]:
        """Generate trajectory by cycling over the files."""
        for file in cycle(self.traj_files):
            # TODO: use the scanner config.
            traj, params = read_trajectory(file, dwell_time=0.01 / self.traj_osf)
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

    __handler_name__ = "acquisition-radial"

    n_shots: int
    n_points: int
    expansion: str = "rotation"
    n_repeat: int = 1
    angle: str = "constant"
    shot_time_ms: int = 20
    smaps: bool = True
    backend: str = "finufft"
    n_jobs: int = 4
    constant: bool = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()

        self.constant = self.angle == "constant"
        self._traj_params |= {
            "n_shots": self.n_shots,
            "n_points": self.n_points,
            "expansion": self.expansion,
            "n_repeat": self.n_repeat,
            "shot_time_ms": self.shot_time_ms,
        }

    def _handle(self, sim: SimData) -> SimData:
        sim.extra_infos["operator"] = self.backend

        return self._acquire(
            sim,
            trajectory_generator=rotate_trajectory(
                trajectory_generator(radial_factory, sim.shape, **self._traj_params),
                self.angle,
            ),
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

    __handler_name__ = "acquisition-sos"

    acsz: float | int
    accelz: int
    orderz: VDSorder = VDSorder.CENTER_OUT
    n_samples: int = 3000
    nb_revolutions: int = 10
    spiral_name: str = "archimedes"
    in_out: bool = True
    pdfz: VDSpdf = VDSpdf.GAUSSIAN
    constant: bool = False
    rotate_angle: AngleRotation = AngleRotation.ZERO
    smaps: bool = True
    backend: str = "finufft"
    shot_time_ms: int = 50
    n_jobs: int = 4

    def __post_init__(self):
        super().__post_init__()
        self._traj_params |= {
            "acsz": self.acsz,
            "accelz": self.accelz,
            "orderz": self.orderz,
            "pdfz": self.pdfz,
            "n_samples": self.n_samples,
            "nb_revolutions": self.nb_revolutions,
            "spiral": self.spiral_name,
            "in_out": self.in_out,
            "rotate_angle": self.rotate_angle,
        }

    def _handle(self, sim: SimData) -> SimData:
        self._traj_params["shape"] = sim.shape
        self._traj_params["rng"] = validate_rng(sim.rng)

        sim.extra_infos["operator"] = "stacked-" + self.backend

        return self._acquire(
            sim,
            trajectory_generator=trajectory_generator(
                stack_spiral_factory, **self._traj_params
            ),
            # extra kwargs for the nufft operator
            z_index="auto",
            backend_name=self.backend,
        )
