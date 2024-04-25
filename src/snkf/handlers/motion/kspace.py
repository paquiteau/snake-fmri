"""Add Motion in the k-space.

TODO: 2D Support
TODO: Actually modify the k-space data

"""

import numpy as np
from numpy.typing import NDArray

from ...simulation import SimData
from ..base import AbstractHandler, requires_field
from .utils import motion_generator, rotation3d


@requires_field(["kspace_data", "kspace_mask"])
class RandomMotionKspaceHandler(AbstractHandler):
    """Add Random Motion in the K-space.

    Parameters
    ----------
    ts_std_mm
        Translation standard deviation, in mm/s.
    rs_std_mm
        Rotation standard deviation, in radians/s.

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

    __handler_name__ = "motion-kspace"

    ts_std_mm: tuple[float, float, float]
    rs_std_mm: tuple[float, float, float]

    def _handle(self, sim: SimData) -> SimData:
        n_frames = sim.data_acq.shape[0]
        # update the translation to be in voxel units
        ts_std_pix = np.array(self.ts_std_mm) / np.array(sim.res_mm)
        motion = motion_generator(
            n_frames,
            ts_std_pix,
            self.rs_std_mm,
            sim.sim_tr,
            sim.rng,
        )
        sim.extra_infos["motion_params"] = motion

        ...


def translate(
    kspace_data: NDArray,
    kspace_loc: NDArray,
    translation_Matrix: NDArray,
) -> NDArray:
    """Translate the image in the kspace data.

    Parameters
    ----------
    kspace_data :(5, length,N)
        the data in the Fourier domain
    kspace_loc : (length,N,3)
        the kspace locations
    translation_Matrix : (length,3)-array
        defines the translation vector in 3d space

    Returns
    -------
    A new kspace_data: (5,N)array
    """
    phi = np.zeros_like(kspace_data)

    for t in range(translation_Matrix.shape[0]):
        for i in range(kspace_loc.shape[2]):
            phi[:, t, :] += kspace_loc[t, :, i] * translation_Matrix[t, i]
    phi = np.reshape(phi, (kspace_data.shape[0], phi.shape[1] * phi.shape[2]))
    return kspace_data * np.exp(-2j * np.pi * phi)


def rotate(kspace_loc_to_corrupt: NDArray, rotation: NDArray) -> NDArray:
    """Rotate the image in the kspace.

    kspace_data :(5, length,N)
        the data in the Fourier domain
    kspace_loc : (length,N,3)
        the kspace locations
    translation : (3,1)-array
        defines the translation vector in 3d space
    td : int
        beginning shot of motion
    tf : int
        final shot of the motion
    center: (3,1) array
        the center of the rotation
    """
    new_loc = np.zeros_like(kspace_loc_to_corrupt)
    for t in range(kspace_loc_to_corrupt.shape[0]):
        R = rotation3d(rotation[:, t])
        new_loc[t, :, :] = np.matmul(kspace_loc_to_corrupt[t, :, :], R)

    return new_loc
