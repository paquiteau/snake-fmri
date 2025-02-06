"""Handler for modifying the Field of View of the Phantom."""

from __future__ import annotations
import warnings
import numpy as np
from numpy.typing import NDArray
import scipy.ndimage as spnd
from scipy.spatial.transform import Rotation as R

from .base import AbstractHandler
from snake.core.parallel import run_parallel
from snake.core.phantom import Phantom
from snake.core.simulation import SimConfig


# TODO allow to use cupy for faster computation (if available)

ThreeInts = tuple[int, int, int]
ThreeFloats = tuple[float, float, float]


def extract_rotated_3d_region(
    volume: NDArray,
    center: ThreeInts,
    size: ThreeInts,
    angles: ThreeFloats,
    zoom_factor: ThreeFloats,
) -> NDArray:
    """
    Extract a rotated 3D rectangular region from a larger 3D array.

    Parameters
    ----------
    volume: np.ndarray
        The 3D source array.
    center: tuple
         The center (x, y, z) of the desired region in the original array.
    size: tuple
        The size (dx, dy, dz) of the extracted region.
    float: tuple
        The rotation angles (rx, ry, rz) in degrees.

    Returns
    -------
    np.ndarray: The extracted 3D region.
    """
    dx, dy, dz = size
    new_shape = tuple(round(s / z) for s, z in zip(size, zoom_factor, strict=True))
    rotation_matrix = R.from_euler("xyz", angles, degrees=True).as_matrix()

    # Generate a coordinate grid for the output block
    X, Y, Z = np.meshgrid(
        *tuple(
            np.linspace(-d / 2, d / 2, s) for d, s in zip(size, new_shape, strict=True)
        ),
        indexing="ij",
    )
    coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

    # Rotate and translate coordinates to match original volume
    transformed_coords = rotation_matrix @ coords
    transformed_coords[0, :] += center[0]
    transformed_coords[1, :] += center[1]
    transformed_coords[2, :] += center[2]

    # Interpolate values at computed coordinates
    extracted_region = spnd.map_coordinates(
        volume,
        transformed_coords,
        order=3,
        mode="nearest",
    )

    return extracted_region.reshape(new_shape)


def _apply_transform(
    input: NDArray,
    output: NDArray,
    i: int,
    center: ThreeInts,
    size: ThreeInts,
    angles: ThreeFloats,
    zoom_factor: ThreeFloats,
) -> None:
    # extract region
    output[i] = extract_rotated_3d_region(input[i], center, size, angles, zoom_factor)
    # resample region


class FOVHandler(AbstractHandler):
    """Handler that update the FOV of the simulation.

    Parameters
    ----------
    center: tuple[float, float, float]
        center of the FOV box in millimeters
    size: tuple[float, float, float]
        size of the FOV box in millimeters
    angle: tuple[float, float, float]
        rotation angle in degrees
    target_resolution: tuple[float, float, float]
        resampling resolution
    """

    __handler_name__ = "fov-select"

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    angles: tuple[float, float, float]
    target_res: tuple[float, float, float]

    def get_static(self, phantom: Phantom, sim_conf: SimConfig) -> Phantom:
        """Modify the FOV of the phantom."""
        old_shape = sim_conf.shape
        old_fov = sim_conf.fov_mm

        # compute the FOV coordinate in voxel units
        if sim_conf.shape != phantom.anat_shape:
            raise ValueError("original Phantom shape and SimConfig shape are different")
        center_vox = tuple(round(self.center[i] / sim_conf.res_mm[i]) for i in range(3))
        size_vox = tuple(round(self.size[i] / sim_conf.res_mm[i]) for i in range(3))
        zoom_factor = tuple(self.target_res[i] / sim_conf.res_mm[i] for i in range(3))
        # Extract the FOV for every tissue
        new_masks = np.zeros(
            (
                phantom.n_tissues,
                *tuple(round(size_vox[i] / zoom_factor[i]) for i in range(3)),
            ),
            dtype=phantom.masks.dtype,
        )
        print("=======", size_vox, zoom_factor)

        run_parallel(
            _apply_transform,
            phantom.masks,
            new_masks,
            parallel_axis=0,
            center=center_vox,
            size=size_vox,
            angles=self.angles,
            zoom_factor=zoom_factor,
        )
        # Create a new phantom with updated masks
        new_phantom = phantom.copy()
        new_phantom.masks = new_masks

        # update the sim_config
        new_shape = new_phantom.anat_shape
        new_fov = tuple(
            new_shape[i] * sim_conf.res_mm[i] / zoom_factor[i] for i in range(3)
        )
        sim_conf.shape = new_shape
        sim_conf.fov_mm = new_fov

        warnings.warn(
            "Shape and FOV of the simulation config were updated."
            f"shape: {old_shape}-> {new_shape} and fov: {old_fov} -> {new_fov}",
            stacklevel=0,
        )
        return new_phantom
