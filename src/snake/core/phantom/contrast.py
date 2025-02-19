"""This module contains functions to compute the contrast of different MRI sequences."""

import numpy as np
from numpy import ndarray as NDArray
from .utils import PropTissueEnum

__all__ = ["_contrast_gre"]


def _contrast_gre(
    props: NDArray,
    *,
    TR: float,
    TE: float,
    FA: float,
) -> NDArray:
    """Compute the GRE contrast for each of the tissues."""
    return (
        props[:, PropTissueEnum.rho],
        *np.sin(np.deg2rad(FA))
        * np.exp(-TE / props[:, PropTissueEnum.T2s])
        * (1 - np.exp(-TR / props[:, PropTissueEnum.T1]))
        / (1 - np.cos(np.deg2rad(FA)) * np.exp(-TR / props[:, PropTissueEnum.T1])),
    )


# TODO Define more contrasts
