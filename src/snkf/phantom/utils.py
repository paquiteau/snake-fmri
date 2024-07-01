"""utilities for phantoms."""

from numpy.typing import NDArray
from snkf.engine.parallel import run_parallel

from scipy.ndimage import zoom
from scipy.ndimage import rotate


def resize_tissues(input, output, i, z, order=3):
    """Resize the tissues."""
    output[i] = zoom(input[i], z, order=order)
