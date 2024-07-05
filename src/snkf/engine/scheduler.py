#!/usr/bin/env python3


import dask.array as da
import dask.distributed as dd
import itertools
import numpy as np
from ..simulation import Phantom, DynamicTissue, SimulationConfig

from .engine import (
    acquire_shots_tissues_nufft,
    acquire_shots_tissues_fft,
)

ACQUISITION_ENGINE = {
    "gpunufft": acquire_shots_tissues_nufft,
    "fft": acquire_shots_tissues_fft,
}


# TODO: Worker that is responsible for a chunk of simulation
# TODO: Specialized worker for specific sampling: Stacked, Cartesian 3D , 3D Nufft
# TODO:
