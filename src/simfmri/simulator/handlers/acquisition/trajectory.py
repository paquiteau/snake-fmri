"""K-spac trajectory data structure."""
from __future__ import annotations

from typing import Literal

import numpy as np

from simfmri.utils import validate_rng
from simfmri.utils.cartesian_sampling import (
    flip2center,
    get_kspace_slice_loc,
)
from simfmri.utils.typing import RngType, Shape2d3d


def accelerate_TR(TR: float, base_TR: float, accel: int) -> float:
    """Compute the TR of an accelerated acquisition."""
    if TR is None and base_TR is None:
        raise ValueError("Either TR or base_TR should be provided.")
    if TR is not None and base_TR is not None:
        raise ValueError("TR and base_TR are exclusive.")
    if TR is None:
        TR = base_TR / accel

    return TR


class KspaceTrajectory:
    """Data structure representing the kspace sampling of one full volume.

    A k-space trajectory is a list of shot, each shot is a list of point.
    Each point is a tuple of (x,y,z) coordinates in the kspace.
    Associated at each point is a sampling time, relative to the beginning of the shot.
    The sampling time is in [0,TR], where TR is the time to acquire a full volume.


    Parameters
    ----------
    n_shots
        Number of shots in the trajectory.
    n_points
        Number of points in each shot.
    is_cartesian
        True if the trajectory lies on a cartesian grid.
    TR
        Time to acquire a full volume.
    dim
        Dimension of the kspace. (2 or 3)


    Attributes
    ----------
    shots
        Array of shape (n_shots, n_points, dim) representing the kspace trajectory.
    sampling_times
        Array of shape (n_shots,2) representing the sampling time window of each shot.
    """

    def __init__(
        self, n_shots: int, n_points: int, is_cartesian: bool, TR: float, dim: int = 3
    ):
        self.is_cartesian = is_cartesian
        self.TR = TR
        self.n_points = n_points

        # sampling time should remain sorted.
        self._sampling_times = np.linspace(0, TR, n_shots, endpoint=False)
        self._shots = np.zeros((n_shots, n_points, dim), dtype=np.float32)

    @property
    def shots(self) -> np.ndarray:
        """Return the array of shots."""
        return self._shots

    @property
    def sampling_times(self) -> np.ndarray:
        """Return the array of sampling times."""
        return self._sampling_times

    @sampling_times.setter
    def sampling_times(self, value: np.ndarray) -> None:
        if (
            np.any(value[:-1] > value[1:])
            or np.any(value < 0)
            or np.any(value > self.TR)
        ):
            raise ValueError("Sampling times should be sorted and in [0,TR].")
        self._sampling_times = value

    @shots.setter
    def shots(self, value: np.ndarray) -> None:
        if self.is_cartesian:
            if not np.all(np.isclose(value, np.round(value))):
                raise ValueError("Cartesian sampling should be on a grid.")
            self._shots = np.round(value).astype(np.int32)
        else:
            self._shots = value

    def extract_trajectory(self, begin: float, end: float) -> KspaceTrajectory:
        """Return a new trajectory, extracted from the current one.

        Parameters
        ----------
        begin
            Beginning of the interval to extract.
        end
            End of the interval to extract.

        Returns
        -------
        new_traj
            New trajectory, extracted from the current one.
        """
        # Find the shots that are in the interval
        begin_shot = np.searchsorted(self._sampling_times, begin, side="right")
        end_shot = np.searchsorted(self._sampling_times, end, side="left")

        new_traj = KspaceTrajectory(
            end_shot - begin_shot, self.n_points, self.is_cartesian, self.TR
        )
        new_traj._sampling_times = self._sampling_times[begin_shot:end_shot]
        new_traj._shots = self._shots[begin_shot:end_shot]

        return new_traj

    def get_binary_mask(self, shape: tuple) -> np.ndarray:
        """
        Return the sampling mask for the given shape, if the trajectory is cartesian.

        Parameters
        ----------
        shape
            Shape of the mask to return.

        Returns
        -------
        mask
            Sampling mask of shape `shape`.

        Raises
        ------
        NotImplementedError
            If the trajectory is not cartesian.
        ValueError
            If the shape has not the same number of dimension as the trajectory.
        """
        if not self.is_cartesian:
            raise NotImplementedError("Non cartesian sampling not implemented.")
        if len(shape) != self._shots.shape[-1]:
            raise ValueError("Shape should be of length %d" % len(self._shots[0]))

        mask = np.zeros(shape, dtype=bool)
        for shot in self._shots:
            mask[shot] = 1
        return mask

    @classmethod
    def vds(
        cls,
        shape: Shape2d3d,
        acs: float | int,
        accel: int,
        accel_axis: int,
        direction: Literal["center-out", "random"],
        TR: float = None,
        base_TR: float = None,
        pdf: Literal["gaussian", "uniform"] = "gaussian",
        rng: RngType = None,
    ) -> KspaceTrajectory:
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
        TR = accelerate_TR(TR, base_TR, accel)

        rng = validate_rng(rng)

        if accel_axis < 0:
            accel_axis = len(shape) + accel_axis
        if not (0 < accel_axis < len(shape)):
            raise ValueError(
                "accel_axis should be lower than the number of spatial dimension."
            )

        line_locs = get_kspace_slice_loc(shape[accel_axis], acs, accel, pdf, rng)
        n_points_shots = np.prod(shape) // shape[accel_axis]
        n_shots = len(line_locs)
        if direction == "center-out":
            line_locs = flip2center(
                sorted(line_locs), np.argmin(line_locs - shape[accel_axis] // 2)
            )
        elif direction == "random":
            line_locs = rng.permutation(line_locs)
        else:
            raise ValueError(f"Unknown direction '{direction}'.")
        # Create the trajectory
        traj = KspaceTrajectory(
            n_shots,
            n_points_shots,
            is_cartesian=True,
            TR=TR,
            dim=len(shape),
        )
        if len(shape) == 2:
            one_shot = np.arange(shape[1 - accel_axis])
        elif len(shape) == 3:
            one_shot1 = np.repeat(
                np.arange(shape[1 - accel_axis]), shape[2 - accel_axis]
            )
            one_shot2 = np.repeat(
                np.arange(shape[2 - accel_axis])[None, :], shape[1 - accel_axis], axis=0
            ).ravel("F")

        traj.shots = np.zeros((n_shots, n_points_shots, len(shape)), dtype=np.int32)
        for shot_idx, line_loc in enumerate(line_locs):
            traj.shots[shot_idx, :, accel_axis] = line_loc
            if len(shape) == 2:
                traj.shots[shot_idx, :, 1 - accel_axis] = one_shot
            elif len(shape) == 3:
                traj.shots[shot_idx, :, 1 - accel_axis] = one_shot1
                traj.shots[shot_idx, :, 2 - accel_axis] = one_shot2
            else:
                raise ValueError("Only 2D and 3D trajectories are supported.")
        return traj

    @classmethod
    def vds_mask(
        cls,
        shape: Shape2d3d,
        acs: float | int,
        accel: int,
        accel_axis: int,
        direction: Literal["center-out", "random"],
        TR: float = None,
        base_TR: float = None,
        pdf: Literal["gaussian", "uniform"] = "gaussian",
        rng: RngType = None,
    ) -> np.ndarray:
        """Get the variable density mask."""
        traj = cls.vds(shape, acs, accel, accel_axis, direction, TR, base_TR, pdf, rng)
        return traj.get_binary_mask(shape)

    @classmethod
    def grappa(
        cls, n_shots: int, n_points: int, TR: float, dim: int = 3
    ) -> KspaceTrajectory:
        """Create a grappa sampling trajectory."""
        raise NotImplementedError("Grappa sampling not implemented.")

    @classmethod
    def poisson_sampling(cls) -> KspaceTrajectory:
        """Create a poisson sampling trajectory. 3D only."""
        raise NotImplementedError("Poisson sampling not implemented.")
