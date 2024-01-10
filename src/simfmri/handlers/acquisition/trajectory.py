"""K-spac trajectory data structure."""
from __future__ import annotations

from typing import Literal, Mapping, Any
from collections.abc import Generator
import numpy as np
from simfmri.utils.typing import RngType, AnyShape

from .cartesian_sampling import get_kspace_slice_loc

from mrinufft.trajectories.trajectory2D import (
    initialize_2D_radial,
    initialize_2D_spiral,
)
from mrinufft.trajectories.tools import stack, rotate


def vds_factory(
    shape: AnyShape,
    acs: float | int,
    accel: int,
    accel_axis: int,
    direction: Literal["center-out", "random"],
    shot_time_ms: int | None = None,
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
    if accel_axis < 0:
        accel_axis = len(shape) + accel_axis
    if not (0 <= accel_axis < len(shape)):
        raise ValueError(
            "accel_axis should be lower than the number of spatial dimension."
        )

    line_locs = get_kspace_slice_loc(shape[accel_axis], acs, accel, pdf, rng, direction)
    # initialize the trajetory. -1 is the default value,
    # and we put the line index in the correct axis (0-indexed)
    shots = -np.ones((len(line_locs), 1, len(shape)), dtype=np.int32)
    for shot_idx, line_loc in enumerate(line_locs):
        shots[shot_idx, :, accel_axis] = line_loc
    return shots


def radial_factory(
    shape: AnyShape,
    n_shots: int,
    n_points: int,
    expansion: str | None = None,
    n_repeat: int = 0,
    **kwargs: Mapping[str, Any],
) -> np.ndarray:
    """Create a radial sampling trajectory."""
    traj_points = initialize_2D_radial(n_shots, n_points)

    if len(shape) == 3:
        if expansion is None:
            raise ValueError("Expansion should be provided for 3D radial sampling.")
        if n_repeat is None:
            raise ValueError("n_repeat should be provided for 3D radial sampling.")
        if expansion == "stacked":
            traj_points = stack(
                traj_points,
                nb_stacks=n_repeat,
            )
        elif expansion == "rotated":
            traj_points = rotate(
                traj_points,
                nb_rotations=n_repeat,
            )
    else:
        raise ValueError("Only 2D and 3D trajectories are supported.")

    return traj_points


def stack_spiral_factory(
    shape: AnyShape,
    accelz: int,
    acsz: int | float,
    n_samples: int,
    nb_revolutions: int,
    shot_time_ms: int | None = None,
    in_out: bool = True,
    directionz: Literal["center-out", "random"] = "center-out",
    pdfz: Literal["gaussian", "uniform"] = "gaussian",
    rng: RngType = None,
) -> np.ndarray:
    """Generate a trajectory of stack of spiral."""
    sizeZ = shape[-1]

    z_index = get_kspace_slice_loc(
        sizeZ, acsz, accelz, pdf=pdfz, rng=rng, direction=directionz
    )

    spiral2D = initialize_2D_spiral(
        Nc=1,
        Ns=n_samples,
        nb_revolutions=nb_revolutions,
        in_out=in_out,
    ).reshape(-1, 2)
    z_kspace = (z_index - sizeZ // 2) / sizeZ
    # create the equivalent 3d trajectory
    nsamples = len(spiral2D)
    nz = len(z_kspace)
    kspace_locs3d = np.zeros((nz, nsamples, 3), dtype=np.float32)
    # TODO use numpy api for this ?
    for i in range(nz):
        kspace_locs3d[i, :, :2] = spiral2D
        kspace_locs3d[i, :, 2] = z_kspace[i]

    return kspace_locs3d.astype(np.float32)


#####################################
# Generators                            #
#####################################


ROTATE_ANGLES = {
    "constant": 0,
    "golden": 2.39996322972865332,  # 2pi(2-phi)
    "golden-mri": 1.941678793,  # 115.15 deg
}


def rotate_trajectory(
    trajectories: Generator[np.ndarray, None, None], theta: str | float = 0
) -> Generator[np.ndarray, None, None]:
    """Incrementally rotate a trajectory.

    Parameters
    ----------
    trajectories:
        Trajectory to rotate.
    """
    if not isinstance(theta, float):
        theta = ROTATE_ANGLES[theta]

    for traj in trajectories:
        if traj.ndim == 2:
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
