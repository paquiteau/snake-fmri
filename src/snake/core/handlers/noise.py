import numpy as np
from copy import deepcopy

from numpy.typing import NDArray
import pandas as pd
from functools import partial

from ..._meta import LogMixin
from ..phantom import Phantom, DynamicData
from ..simulation import SimConfig
from .base import AbstractHandler


def apply_noise(phantom: Phantom, data, idx, snr) -> Phantom:
    rng = np.random.default_rng((int(data[0, idx]), idx))  # new seed for every frame
    noise_tissue = rng.standard_normal(size=phantom.masks.shape[1:]) / snr
    new_phantom = deepcopy(phantom)
    tissue_idx = list(phantom.labels).index("noise")
    new_phantom.masks[tissue_idx] = phantom.masks[tissue_idx] * noise_tissue
    return new_phantom


class NoiseHandler(AbstractHandler):

    __handler_name__ = "noise-image"
    snr: float

    def get_static(self, phantom: Phantom, sim_conf: SimConfig) -> Phantom:
        """Add a static noise tissue"""
        noise_tissue = np.ones_like(phantom.masks[0])
        noise_props = np.array([[100000, 100000, 100000, 1, 0]])
        new_phantom = phantom.add_tissue(
            "noise", noise_tissue, noise_props, phantom.name + "-noise"
        )

        return new_phantom

    def get_dynamic(self, phantom: Phantom, sim_conf: SimConfig) -> DynamicData:
        """Add a dynamic noise tissue"""
        return DynamicData(
            "noise",
            data=np.ones((1, sim_conf.max_n_shots), dtype=np.int32) * sim_conf.rng_seed,
            func=partial(apply_noise, snr=self.snr),
        )
