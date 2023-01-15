"""Cartesian sampling simulation."""

import numpy as np
from scipy.stats import norm
from simfmri.utils import RngType, validate_rng


def get_kspace_slice_loc(
    dim_size: int,
    center_prop: int | float,
    accel: int = 4,
    pdf: str = "gaussian",
    rng: RngType = None,
) -> np.ndarray:
    """Get slice index at a random position.

    Parameters
    ----------
    dim_size: int
        Dimension size
    center_prop: float or int
        Proportion of center of kspace to continuouly sample
    accel: float
        Undersampling/Acceleration factor
    pdf: str, optional
        Probability density function for the remaining samples.
        "gaussian" (default) or "uniform".
    rng: random state

    Returns
    -------
    np.ndarray: array of size dim_size/accel.
    """
    indexes = list(range(dim_size))
    if isinstance(center_prop, int):
        center_prop = center_prop / dim_size

    center_start = int(dim_size * (0.5 - center_prop / 2))
    center_stop = int(dim_size * (0.5 + center_prop / 2))

    center_indexes = indexes[center_start:center_stop]
    borders = np.asarray([*indexes[:center_start], *indexes[center_stop:]])

    n_samples_borders = int((dim_size - len(center_indexes)) / accel)
    if n_samples_borders < 1:
        raise ValueError(
            "acceleration factor, center_prop and dimension not compatible."
            "Edges will not be sampled. "
        )
    rng = validate_rng(rng)

    if pdf == "gaussian":
        p = norm.pdf(np.linspace(norm.ppf(0.001), norm.ppf(0.999), len(borders)))
    elif pdf == "uniform":
        p = np.ones(len(borders))
    else:
        raise ValueError("Unsupported value for pdf.")
        # TODO: allow custom pdf as argument (vector or function.)

    p /= np.sum(p)
    sampled_in_border = list(
        rng.choice(borders, size=n_samples_borders, replace=False, p=p)
    )

    return np.array(sorted(center_indexes + sampled_in_border))


def get_cartesian_mask(
    shape: tuple,
    n_frames: int,
    rng: RngType = None,
    constant: bool = False,
    center_prop: float | int = 0.3,
    accel: int = 4,
    accel_axis: int = 0,
    pdf: str = "gaussian",
) -> np.ndarray:
    """
    Get a cartesian mask for fMRI kspace data.

    Parameters
    ----------
    shape: tuple
        shape of fMRI volume.
    n_frames: int
        number of frames.
    rng: Generator or int or None (default)
        Random number generator or seed.
    constant: bool
        If True, the mask is constant across time.
    center_prop: float
        Proportion of center of kspace to continuouly sample
    accel: float
        Undersampling/Acceleration factor
    pdf: str, optional
        Probability density function for the remaining samples.
        "gaussian" (default) or "uniform".
    rng: random state

    Returns
    -------
    np.ndarray: random mask for an acquisition.
    """
    rng = validate_rng(rng)

    mask = np.zeros((n_frames, *shape))
    slicer = [slice(None, None, None)] * (1 + len(shape))
    if accel_axis < 0:
        accel_axis = len(shape) + accel_axis
    if not (0 < accel_axis < len(shape)):
        raise ValueError(
            "accel_axis should be lower than the number of spatial dimension."
        )
    if constant:
        mask_loc = get_kspace_slice_loc(shape[accel_axis], center_prop, accel, pdf, rng)
        slicer[accel_axis + 1] = mask_loc
        mask[tuple(slicer)] = 1
        return mask

    for i in range(n_frames):
        mask_loc = get_kspace_slice_loc(shape[accel_axis], center_prop, accel, pdf, rng)
        slicer[0] = i
        slicer[accel_axis + 1] = mask_loc
        mask[tuple(slicer)] = 1
    return mask
