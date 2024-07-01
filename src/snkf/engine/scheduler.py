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


def get_all_kspace(
    sim_conf: SimulationConfig,
    backend="gpunufft",
    n_workers=1,
    threads_per_worker=1,
    shot_per_task=1,
):
    # setup client and cluster

    client = dd.Client(n_workers=n_workers, threads_per_worker=threads_per_worker)

    acquisition = ACQUISITION_ENGINE[backend]

    # move to dask
    for shot in sim_conf.shots:
        #
        pass


def get_all_image():
    # setup dask
    pass
