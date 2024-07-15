"""Utilities for the MRD format."""

import base64
import pickle
from enum import IntFlag
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..phantom import Phantom, PropTissueEnum
from ..simulation import SimConfig


def get_contrast_gre(
    phantom: Phantom, FA: NDArray, TE: NDArray, TR: NDArray
) -> NDArray:
    """Compute the GRE contrast at TE."""
    return (
        phantom.tissue_properties[:, PropTissueEnum.rho]
        * np.sin(np.deg2rad(FA))
        * np.exp(-TE / phantom.tissue_properties[:, PropTissueEnum.T2s])
        * (1 - np.exp(-TR / phantom.tissue_properties[:, PropTissueEnum.T1]))
        / (
            1
            - np.cos(np.deg2rad(FA))
            * np.exp(-TR / phantom.tissue_properties[:, PropTissueEnum.T1])
        )
    )


def get_ideal_phantom(phantom: Phantom, sim_conf: SimConfig) -> NDArray:
    """Apply the contrast to the phantom and return volume."""
    contrast = get_contrast_gre(
        phantom, sim_conf.seq.FA, sim_conf.seq.TE, sim_conf.seq.TR
    )
    print(contrast)
    phantom_state = np.sum(
        phantom.tissue_masks * contrast[(..., *([None] * len(phantom.anat_shape)))],
        axis=0,
    )
    return phantom_state
