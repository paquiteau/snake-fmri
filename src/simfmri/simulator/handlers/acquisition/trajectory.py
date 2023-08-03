"""K-spac trajectory data structure."""
from __future__ import annotations

from typing import Literal, Callable, Generator
import numpy as np
import logging
from simfmri.utils import validate_rng
from simfmri.utils.typing import RngType, Shape2d3d

from .cartesian_sampling import (
    flip2center,
    get_kspace_slice_loc,
)

logger = logging.getLogger("simulation.acquisition.trajectory")


def vds(
    shape: Shape2d3d,
    acs: float | int,
    accel: int,
    accel_axis: int,
    direction: Literal["center-out", "random"],
    shot_time_ms: int = None,
    pdf: Literal["gaussian", "uniform"] = "gaussian",
    rng: RngType = None,
) -> np.ndarray:
    """
    Create a variable density sampling trajectory.

    Parameters
    ----------
    shape
        Shape of the kspace.
    acs
        autocalibration line number (int) or proportion (float)
    direction
        Direction of the sampling.
    TR
        Time to acquire the k-space. Exclusive with base_TR.
    base_TR
        Time to acquire a full volume in the base trajectory. Exclusive with TR.
    pdf
        Probability density function of the sampling. "gaussian" or "uniform"
    rng
        Random number generator or seed.

    Returns
    -------
    KspaceTrajectory
        Variable density sampling trajectory.
    """
    rng = validate_rng(rng)
    if accel_axis < 0:
        accel_axis = len(shape) + accel_axis
    if not (0 <= accel_axis < len(shape)):
        raise ValueError(
            "accel_axis should be lower than the number of spatial dimension."
        )

    line_locs = get_kspace_slice_loc(shape[accel_axis], acs, accel, pdf, rng)
    n_points_shots = np.prod(shape) // shape[accel_axis]
    n_shots = len(line_locs)
    if direction == "center-out":
        line_locs = flip2center(sorted(line_locs), shape[accel_axis] // 2)
    elif direction == "random":
        line_locs = rng.permutation(line_locs)
    elif direction is None:
        pass
    else:
        raise ValueError(f"Unknown direction '{direction}'.")

    shots = np.zeros((n_shots, n_points_shots, len(shape)), dtype=np.int32)
    for shot_idx, line_loc in enumerate(line_locs):
        shots[shot_idx, :, accel_axis] = line_loc
    return shots


def radial(
    n_shots: int,
    n_points: int,
    dim: Literal[2, 3] = 2,
    expansion: str = None,
    n_repeat: int = None,
    TR_ms: int = None,
    shot_time_ms: int = None,
) -> np.ndarray:
    """Create a radial sampling trajectory."""
    from mrinufft.trajectories.trajectory2D import initialize_2D_radial
    from mrinufft.trajectories.trajectory3D import initialize_3D_from_2D_expansion

    if dim == 2:
        traj_points = initialize_2D_radial(n_shots, n_points)
        traj_points = np.float32(traj_points)

    elif dim == 3:
        if expansion is None:
            raise ValueError("Expansion should be provided for 3D radial sampling.")
        if n_repeat is None:
            raise ValueError("n_repeat should be provided for 3D radial sampling.")
        traj_points = initialize_3D_from_2D_expansion(
            basis="radial",
            expansion=expansion,
            Nc=n_shots,
            Ns=n_points,
            nb_repetitions=n_repeat,
        )
    else:
        raise ValueError("Only 2D and 3D trajectories are supported.")

    return traj_points


def trajectory_generator(
    traj_factory: Callable, **kwargs: None
) -> Generator[np.ndarray]:
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
    while True:
        yield traj_factory(**kwargs)


ROTATE_ANGLES = {
    "constant": 0,
    None: 0,
    "golden": 2.39996322972865332,  # 2pi(2-phi)
    "golden-mri": 1.941678793,  # 115.15 deg
}


def rotate_trajectory(
    trajectories: Generator[np.ndarray], theta: float | str = None
) -> np.ndarray:
    """Incrementally rotate a trajectory.

    Parameters
    ----------
    trajectories:
        Trajectory to rotate.
    """
    if not isinstance(theta, float):
        theta = ROTATE_ANGLES[theta]

    for traj in trajectories:
        if traj.dim == 2:
            rot = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
        else:
            rot = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )

        theta += theta

        yield np.einsum("ij,klj->kli", rot, traj)


def kspace_bulk_shot(
    traj_generator: Generator[np.ndarray],
    n_batch: int = 3,
) -> Generator[tuple[np.ndarray, list[int]]]:
    """Generate a stream of shot, delivered in batch.

    Parameters
    ----------
    traj_factory: Callable
        A function that create a ndarray representing a kspace trajectory
    n_batch: int
        The number of shots delivered together. Typically n_batch*shot_time = sim_time
        (time for a single sim_frame)
    factory_kwargs: dict
        Parameter for the trajectory

    Yields
    ------
    tuple[np.ndarray, list[int]]
        A tuple of (shots, kspace_frame) where kspace_frame is the index of the kspace
        frame in the full trajectory.
    """
    shots = next(traj_generator)
    shot_idx = 0
    print(f"full trajectory has {len(shots)} shots")
    kspace_frame = 0
    while True:
        if shot_idx + n_batch <= len(shots):
            yield shots[shot_idx : shot_idx + n_batch], [kspace_frame] * n_batch
            shot_idx += n_batch
        elif shot_idx <= len(shots):
            # The last batch is incomplete, so we start a new trajectory
            # to complete it.
            new_shots = next(traj_generator)
            new_shot_idx = shot_idx + n_batch - len(shots)
            yield (
                np.vstack([shots[shot_idx:], new_shots[:new_shot_idx]]),
                [kspace_frame] * (len(shots) - shot_idx)
                + [kspace_frame + 1] * new_shot_idx,
            )
            shots = new_shots
            shot_idx = new_shot_idx
            kspace_frame += 1
        else:
            raise RuntimeError("invalid shot_idx value")
