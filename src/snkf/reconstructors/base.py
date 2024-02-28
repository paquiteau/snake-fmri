"""Base Interfaces for the reconstructors."""

import logging
from dataclasses import field
import numpy as np
from typing import Protocol, Any, ClassVar
from snkf.simulation import SimData
from snkf.base import MetaDCRegister

logger = logging.getLogger("Reconstructor")


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


class MetaReconstructor(MetaDCRegister):
    """MetaClass Reconstructor."""

    dunder_name = "reconstructor"


class BaseReconstructor(metaclass=MetaReconstructor):
    """Represents the interface required to be benchmark-able."""

    __registry__: ClassVar[dict]
    __reconstructor_name__: ClassVar[str]

    nufft_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        pass

    def setup(self, sim: SimData) -> None:
        """Set up the reconstructor."""
        logger.info(f"Setup reconstructor {self.__class__.__name__}")

    def reconstruct(self, sim: SimData) -> np.ndarray:
        """Reconstruct data."""
        raise NotImplementedError()


def get_reconstructor(name: str) -> type[BaseReconstructor]:
    """Get a handler from its name."""
    try:
        return BaseReconstructor.__registry__[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown Reconstructor, {name}, available are {list_reconstructors()}"
        ) from e


def list_reconstructors() -> list[str]:
    """List available reconstructors."""
    return list(BaseReconstructor.__registry__.keys())
