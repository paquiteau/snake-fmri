"""utilities for phantoms."""

from numpy.typing import NDArray

from scipy.ndimage import zoom


def resize_tissues(
    input: NDArray, output: NDArray, i: int, z: tuple[float], order: int = 3
) -> None:
    """Resize the tissues."""
    output[i] = zoom(input[i], z, order=order)
