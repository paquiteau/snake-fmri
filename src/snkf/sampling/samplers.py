"""Samplers generate kspace trajectories."""

import numpy as np
from numpy.typing import NDArray

from ..phantom import Phantom
from ..simulation import SimConfig
from .base import BaseSampler
from .factories import (
    AngleRotation,
    VDSorder,
    VDSpdf,
    stack_spiral_factory,
)


class StackOfSpiralSampler(BaseSampler):
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
    obs_ms: int
        Time spent to acquire a single shot
    nb_revolutions: int
        Number of revolutions of the spiral.
    in_out: bool
        If true, the spiral is acquired with a double join pattern from/to the periphery
    **kwargs:
        Extra arguments (smaps, n_jobs, backend etc...)
    """

    __sampler_name__ = "stack-of-spiral"

    acsz: float | int
    accelz: int
    orderz: VDSorder = VDSorder.CENTER_OUT
    nb_revolutions: int = 10
    spiral_name: str | float = "archimedes"
    pdfz: VDSpdf = VDSpdf.GAUSSIAN
    constant: bool = False
    rotate_angle: AngleRotation = AngleRotation.ZERO
    obs_time_ms: int = 30
    n_shot_slices: int = 1

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate the sampling pattern."""
        n_samples = int(self.obs_time_ms / sim_conf.hardware.dwell_time_ms)
        trajectory = stack_spiral_factory(
            shape=sim_conf.shape,
            accelz=self.accelz,
            acsz=self.acsz,
            n_samples=n_samples,
            nb_revolutions=self.nb_revolutions,
            pdfz=self.pdfz,
            orderz=self.orderz,
            spiral=self.spiral_name,
            rotate_angle=self.rotate_angle,
            in_out=self.in_out,
            n_shot_slices=self.n_shot_slices,
            rng=sim_conf.rng,
        )
        self._n_shot_frames = trajectory.shape[0]
        self._n_samples_shot = trajectory.shape[1]
        return trajectory


class EPI3dAcquisitionSampler(BaseSampler):
    """Sampling pattern for EPI-3D."""

    __sampler_name__ = "epi-3d"
    is_cartesian = True
    in_out = True

    acs_plane: float | int = 0.1
    acs_slice: float | int = 0.1

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate the sampling pattern."""

        kspace_mask = np.zeros(sim_conf.shape, dtype=int)
