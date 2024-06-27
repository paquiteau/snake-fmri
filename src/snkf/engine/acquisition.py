#!/usr/bin/env python3
r"""Engine That perform the acquisition of MRI data using the signal model."""

from mrinufft import get_operator
import dask.array as da
import dask.distributed as dd
import numpy as np


def acquire_shots_tissues_nufft(shots:np.ndarray, tissues:np.ndarray, contrast:np.ndarray, smaps:np.ndarray, t2s_relax, backend="gpunufft"):
    """Acquire a group of shots using the signal model."""


    if len(shots.shape) == 2:
        shots = shots[None, ...] # add batch dimension
        #
    get_operator(backend)(shots, tissues, contrast, t2s_relax)
    for shot in shots:
        # update shot
        # acquire k-space data
        # move to host
        #

def acquire_shots_tissues_stacked_nufft(shots, tissues, contrast, t2s_relax, backend="stacked-gpunufft"):
    """Acquire a group of shots using the signal model."""

    pass


def acquire_shots_tissues_fft(shots, tissues, contrast, t2s_relax):
    """Acquire a group of shots using the signal model."""

    pass
