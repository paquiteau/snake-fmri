"""Add motion in the image domain."""

import numpy as np

from numpy.typing import NDArray

from ...simulation import SimData
from ..base import AbstractHandler, requires_field
from .utils import apply_shift, apply_rotation_at_center, motion_generator

requires_data_acq = requires_field("data_acq", lambda sim: sim.data_ref.copy())


@requires_data_acq
class RandomMotionImageHandler(AbstractHandler):
    """Add Random Motion in Image.

    Parameters
    ----------
    ts_std_mm
        Translation standard deviation, in mm/s.
    rs_std_mm
        Rotation standard deviation, in radians/s.

    motion_file: str
        If provided, the motion file is loaded and resampled to match the number
        of frames in the simulation. The motion is then added to the data.
    motion_file_tr: float
        Original TR of the motion file, in seconds.

    Notes
    -----
    The motion is generated by drawing from a normal distribution with standard
    deviation for the 6 motion parameters (3 translations and 3 rotations, in
    this order). Then the cumulative motion is computed by summing the motion
    at each frame.

    The handlers is parametrized with speed in mm/s and rad/s, as these values
    provides an independent control of the motion amplitude regardless of the
    time resolution for the simulation.
    """

    __handler_name__ = "motion-image"

    ts_std_mms: tuple[float, float, float] | None = None
    rs_std_mms: tuple[float, float, float] | None = None

    motion_file: str | None = None
    motion_file_tr: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if (self.ts_std_mms is None or self.rs_std_mms is None) and (
            self.motion_file is None or self.motion_file_tr is None
        ):
            raise ValueError(
                "At least one of ts_std_mm, rs_std_mm or motion_file must be provided."
            )
        self._motion_data = None
        if self.motion_file is not None:
            # load the motion file
            self._motion_data = np.loadtxt(self.motion_file)

    def _handle(self, sim: SimData) -> SimData:
        n_frames = sim.data_acq.shape[0]
        # update the translation to be in voxel units
        ts_std_pix = np.array(self.ts_std_mms) / np.array(sim.res_mm)

        if self._motion_data is not None:
            # resample the motion data to match the simulation framerate.
            motion = np.interp(
                np.arange(n_frames) * sim.sim_tr,
                np.arange(len(self._motion_data)) * self.motion_file_tr,
                self._motion_data,
            )

        else:
            motion = motion_generator(
                n_frames,
                ts_std_pix,
                self.rs_std_mms,
                sim.sim_tr,
                sim.rng,
            )
        sim.extra_infos["motion_params"] = motion

        if sim.lazy:
            sim.data_acq.apply(add_motion_to_frame, motion)
        else:
            for i in range(len(sim.data_acq)):
                sim.data_acq[i] = add_motion_to_frame(sim.data_acq[i], motion, i)
        return sim


def add_motion_to_frame(
    data: NDArray[np.complexfloating] | NDArray[np.floating],
    motion: NDArray[np.floating],
    frame_idx: int = 0,
) -> np.ndarray:
    """Add motion to a base array.

    Parameters
    ----------
    data: np.ndarray
        The data to which motion is added.
    motion: np.ndarray
        The N_frames x 6 motion trajectory.
    frame_idx: int
        The frame index used to compute the motion at that frame.

    Returns
    -------
    np.ndarray
        The data with motion added.
    """
    rotated = apply_rotation_at_center(data, motion[frame_idx, 3:])
    rotated_and_translated = apply_shift(rotated, motion[frame_idx, :3])
    return rotated_and_translated
