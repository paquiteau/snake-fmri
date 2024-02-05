"""Base Interfaces for the reconstructors."""

import logging
import numpy as np
from typing import Protocol
from snkf.simulation import SimData

logger = logging.getLogger("Reconstructor")
RECONSTRUCTORS = {}


class SpaceFourierProto(Protocol):
    """Fourier operator interface."""

    n_frames: int
    shape: tuple[int]
    n_coils: int
    uses_sense: bool

    def op(self, x: np.ndarray) -> np.ndarray:
        """Apply the Fourier operator."""
        ...

    def adj_op(self, x: np.ndarray) -> np.ndarray:
        """Apply the adjoint of the Fourier operator."""
        ...


class BaseReconstructor:
    """Represents the interface required to be benchmark-able."""

    name: None | str = None
    fourier_op: SpaceFourierProto

    def __init__(self, nufft_kwargs: dict | None = None):
        self.reconstructor = None
        self.nufft_kwargs = nufft_kwargs or {}

    def __init_subclass__(cls):
        """Register reconstructors."""
        if cls_name := getattr(cls, "name", None):
            RECONSTRUCTORS[cls_name] = cls

    def setup(self, sim: SimData) -> None:
        """Set up the reconstructor."""
        logger.info(f"Setup reconstructor {self.__class__.__name__}")

    def reconstruct(self, sim: SimData) -> np.ndarray:
        """Reconstruct data."""
        raise NotImplementedError()

    def __str__(self):
        return self.name


def get_reconstructor(name: str) -> type[BaseReconstructor]:
    """Get a handler from its name."""
    try:
        return RECONSTRUCTORS[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown Reconstructor, {name}, available are {list_reconstructors()}"
        ) from e


def list_reconstructors() -> list[str]:
    """List available reconstructors."""
    return list(RECONSTRUCTORS.keys())
