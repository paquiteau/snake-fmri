"""Interface to BrainWeb data set."""

import numpy as np
from scipy.spatial.transform import Rotation


def get_indices_inside_ellipsoid(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
    semi_axes_lengths: tuple[int, int, int],
    euler_angles: tuple[int, int, int],
) -> np.ndarray:
    # Create an array with the specified shape
    indices = np.indices(shape).reshape(3, -1).T
    # Shift indices to center
    shifted_indices = indices - np.array(center)
    # Apply rotation to align with ellipsoid orientation
    rotation = Rotation.from_euler("zyx", euler_angles, degrees=True)
    rotated_indices = rotation.apply(shifted_indices)
    # Normalize the indices
    normalized_indices = rotated_indices / semi_axes_lengths
    # Calculate the distance from the origin for each index
    distances = np.linalg.norm(normalized_indices, axis=1)
    # Find the indices that lie inside the ellipsoid
    # inside_indices = np.where(distances <= 1)
    distances = np.reshape(distances, shape)
    return distances <= 1


# this define a ellipsoid roi located in the occipital cortex.
BRAINWEB_OCCIPITAL_ROI = {
    "shape": (362, 434, 362),
    "center": (185, 52, 145),
    "semi_axes_lengths": (100, 20, 50),
    "euler_angles": (0, 0, -5),
}
