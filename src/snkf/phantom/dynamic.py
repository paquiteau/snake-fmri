"""Dynamic data object."""

from __future__ import annotations
from dataclasses import dataclass
from numpy.typing import NDArray
from collections.abc import Callable, Sequence
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

    @classmethod
    def from_mrd_dataset(cls, dataset: str, chunk: Sequence[int]) -> DynamicData:
        """Create DynamicData from dataset."""
        # TODO Create DynamicData from mrd.
        return cls(None, None)
