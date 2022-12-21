"""Cartesian sampling simulation."""
from typing import Union

import numpy as np
from scipy.stats import norm
from simfmri.utils import RngType, validate_rng


def get_kspace_slice_loc(
    dim_size: int,
    center_prop: Union(int, float),
    accel: int = 4,
    pdf: str = "gaussian",
    rng: RngType = None,
):
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

    n_samples_borders = int((dim_size / accel) - len(center_indexes))
    if n_samples_borders < 1:
        raise ValueError(
            "acceleration factor, center_prop and dimension not compatible."
            "Edges will not be sampled. "
        )
    rng = validate_rng(rng)

    if pdf == "gaussian":
        p = norm.pdf(np.linspace(norm.ppf(0.001), norm.ppf(0.999), len(borders)))
        p /= np.sum(p)
    elif pdf == "uniform":
        p = np.ones(len(borders)) / len(borders)
    else:
        raise ValueError("Unsupported value for pdf.")
    # TODO:
    # allow custom pdf as argument (vector or function.)

    sampled_in_border = list(rng.choice(borders, size=n_samples_borders, replace=False))

    return np.array(sorted(center_indexes + sampled_in_border))


def get_cartesian_mask(
    shape: tuple,
    n_frames: int,
    rng: RngType = None,
    constant: bool = False,
    center_prop: Union(float, int) = 0.3,
    accel: int = 4,
    pdf: str = "gaussian",
):
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
    if constant:
        mask_loc = get_kspace_slice_loc(shape[-1], center_prop, accel, pdf, rng)
        mask[:, mask_loc] = 1
        return mask

    for i in range(n_frames):
        mask_loc = get_kspace_slice_loc(shape[-1], center_prop, accel, pdf, rng)
        mask[i, mask_loc] = 1
    return mask
