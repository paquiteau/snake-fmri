"""Base Acquisition handlers."""

from __future__ import annotations

from typing import Any, Literal, Mapping
from collections.abc import Generator

import numpy as np


from simfmri.simulator.handlers.base import AbstractHandler
from simfmri.simulator.simulation import SimDataType
from simfmri.utils import validate_rng

from .trajectory import (
    vds_factory,
    radial_factory,
    stack_spiral_factory,
    TrajectoryGeneratorType,
    trajectory_generator,
    rotate_trajectory,
)

from ._coils import get_smaps
from .workers import acquire_cartesian_mp, acquire_noncartesian_mp


SimGeneratorType = Generator[np.ndarray, None, None]


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

    acquire_mp = staticmethod(acquire_cartesian_mp)

    def __init__(self, constant: bool, smaps: bool):
        super().__init__()
        self.constant = constant
        self.smaps = smaps
        self.is_cartesian = True

    def _acquire(
        self,
        sim: SimDataType,
        trajectory_generator: TrajectoryGeneratorType,
        **kwargs: Mapping[str, Any],
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
        n_shot_traj = len(test_traj)
        shot_time_ms = self._traj_params["shot_time_ms"]
        if sim.sim_tr_ms % shot_time_ms != 0:
            # find the closest shot time that divides the simulation time
            new_shot_time_ms = int(
                sim.sim_tr_ms / ((sim.sim_tr_ms // shot_time_ms) + 1)
            )
            self.log.warning(
                f"shot time {shot_time_ms}ms does not divide sim tr {sim.sim_tr_ms}ms."
                f" Updating to {new_shot_time_ms}ms per shot"
            )
            shot_time_ms = new_shot_time_ms

        kframe_tr = n_shot_traj * shot_time_ms
        self.log.debug("initial TR volume %i ms", kframe_tr)

        n_shot_sim_frame = int(sim.sim_tr_ms / shot_time_ms)
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

    """

    name = "acquisition-vds"

    def __init__(
        self,
        acs: float | int,
        accel: int,
        accel_axis: int,
        direction: Literal["center-out", "random"],
        shot_time_ms: int = None,
        pdf: Literal["gaussian", "uniform"] = "gaussian",
        constant: bool = False,
        smaps: bool = True,
    ):
        super().__init__(constant, smaps)
        self._traj_params = {
            "acs": acs,
            "accel": accel,
            "accel_axis": accel_axis,
            "direction": direction,
            "pdf": pdf,
            "shot_time_ms": shot_time_ms,
        }

    def _handle(self, sim: SimDataType) -> SimDataType:
        self._traj_params["shape"] = sim.shape
        self._traj_params["rng"] = validate_rng(sim.rng)
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

    acquire_mp = staticmethod(acquire_noncartesian_mp)

    # TODO: add other trajectories sampling and refactor.
    def __init__(
        self,
        constant: bool = False,
        smaps: bool = True,
        backend: str = "finufft",
    ):
        super().__init__(constant, smaps)
        self._backend = backend


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
        **kwargs: Mapping(str, Any),
    ) -> None:
        super().__init__(constant=angle == "constant", **kwargs)

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
        sim.extra_infos["operator"] = self._backend

        return self._acquire(
            sim,
            trajectory_generator=rotate_trajectory(
                trajectory_generator(radial_factory, **self._traj_params),
                self._angle,
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

    name = "acquisition-sos"

    def __init__(
        self,
        acsz: float | int,
        accelz: int,
        directionz: Literal["center-out", "random"],
        shot_time_ms: int = 50,
        n_samples: int = 3000,
        nb_revolutions: int = 10,
        in_out: bool = True,
        pdfz: Literal["gaussian", "uniform"] = "gaussian",
        **kwargs: Mapping[str, Any],
    ):
        super().__init__(**kwargs)
        self._traj_params = {
            "acsz": acsz,
            "accelz": accelz,
            "directionz": directionz,
            "pdfz": pdfz,
            "shot_time_ms": shot_time_ms,
            "n_samples": n_samples,
            "nb_revolutions": nb_revolutions,
            "in_out": in_out,
        }

    def _handle(self, sim: SimDataType) -> SimDataType:
        self._traj_params["shape"] = sim.shape
        self._traj_params["rng"] = validate_rng(sim.rng)

        sim.extra_infos["operator"] = self._backend

        from mrinufft.trajectories.utils import (
            compute_gradients,
            _check_gradient_constraints,
            DEFAULT_RASTER_TIME_MS,
        )

        tobs = DEFAULT_RASTER_TIME_MS * self._traj_params["n_samples"]
        shot_time = self._traj_params["shot_time_ms"]
        if tobs > shot_time - 20:
            self.log.warning(
                f"The simulated trajectory takes {tobs}ms to run, "
                f" in a {shot_time}ms shot."
            )

        test_traj = stack_spiral_factory(**self._traj_params)
        grads, _, slews = compute_gradients(test_traj, resolution=sim.res[-1])
        if not _check_gradient_constraints(grads, slews):
            self.log.error("Trajectory does not respect gradient constraints.")

        self._traj_params["constant"] = self.constant

        return self._acquire(
            sim,
            trajectory_generator=trajectory_generator(
                stack_spiral_factory, **self._traj_params
            ),
            # extra kwargs for the nufft operator
            op_backend="stacked",
            z_index="auto",
            backend="finufft",
        )
