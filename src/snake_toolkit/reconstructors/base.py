"""Base Class for Reconstructors."""

import logging
from dataclasses import field
from typing import Any, ClassVar
from snake._meta import MetaDCRegister

from snake.simulation import SimConfig


class MetaReconstructor(MetaDCRegister):
    """MetaClass Reconstructor."""

    dunder_name = "reconstructor"


class BaseReconstructor(metaclass=MetaReconstructor):
    """Represents the interface required to be benchmark-able."""

    __registry__: ClassVar[dict]
    __reconstructor_name__: ClassVar[str]
    log: ClassVar[logging.Logger]

    nufft_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        pass

    def setup(self, sim: SimConfig) -> None:
        """Set up the reconstructor."""
        self.log.info(f"Setup reconstructor {self.__class__.__name__}")

    def reconstruct(self, sim: SimConfig) -> np.ndarray:
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
