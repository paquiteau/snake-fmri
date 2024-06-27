
from dataclasses import dataclass
from typing import NamedTuple
from functools import partial, lru_cache
import numpy as np



class ParamsGRE(NamedTuple):
    TR: float
    TE: float
    FA: float

class ParamsHardware(NamedTuple):
    gmax: float
    smax: float
    dwell_time: float
    n_coils: int



@dataclass
class Phantom:
    tissue_masks: np.ndarray
    T1: np.ndarray
    T2: np.ndarray
    T2s: np.ndarray
    rho: np.ndarray
    chi: np.ndarray

    @lru_cache(10)
    def get_contrast_gre(self, FA, TE, TR):
        return np.sin(FA) * np.exp(-TE / self.T2s) * (1 - np.exp(-TR / self.T1)) / (1 - np.cos(FA) * np.exp(-TR / self.T1))


    @classmethod
    def from_brainweb(id, resolution):
        from brainweb_dl import get_mri

        pass

    def from_shepp_logan(resolution):
        pass

    def from_guerin_kern(resolution):
        pass

@dataclass
class DynamicTissue:
    weights: np.ndarray

    def get_contrast(self, i:int):
        raise NotImplementedError


@dataclass
class SimulationConfig:
    phantom: Phantom
    extra_tissues: list[DynamicTissue]
    sim_tr_ms: float
    sequence_params: ParamsGRE
    hardware: ParamsHardware
    shots_loc: np.ndarray
    is_cartesian: bool = True

    @property
    def shape(self):
        return self.phantom.tissue_mask.shape[1:]


    def get_next_state(self, ) -> Simulation:
        return
