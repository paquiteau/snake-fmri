#!/usr/bin/env python3
from dataclasses import dataclass
from numpy.typing import NDArray
from collections.abc import Callable
from .static import Phantom
from ..simulation import SimConfig


@dataclass
class DynamicData:
    """Dynamic data object."""

    data: NDArray
    func: Callable[[NDArray, Phantom], Phantom]

    def apply(self, phantom: Phantom, sim_conf: SimConfig) -> Phantom:
        """Apply the dynamic data to the phantom."""
        return self.func(self.data, phantom)
