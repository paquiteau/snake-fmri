"""Dynamic data object."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray

from ..simulation import SimConfig
from .static import Phantom

log = logging.getLogger(__name__)


@dataclass
class DynamicData:
    """Dynamic data object."""

    name: str
    data: NDArray
    func: Callable[[Phantom, NDArray, int], Phantom]
    in_kspace: bool = False

    def apply(self, phantom: Phantom, sim_conf: SimConfig, time_idx: int) -> Phantom:
        """Apply the dynamic data to the phantom."""
        return self.func(phantom, self.data, time_idx)

    @classmethod
    def _from_waveform(cls, waveform: mrd.Waveform, wave_info: dict) -> DynamicData:
        return DynamicData(
            name=wave_info["name"],
            data=waveform.data.view(np.float32).reshape(
                waveform.channels, waveform.number_of_samples
            ),
            func=wave_info[wave_info["name"]],
            in_kspace=wave_info["domain"] == "kspace",
        )


class KspaceDynamicData(DynamicData):
    """Dynamic DAta that will be applied in the k-space."""

    name: str
    data: NDArray
    func: Callable[[NDArray, NDArray, int], Phantom]
    in_kspace: bool = True
