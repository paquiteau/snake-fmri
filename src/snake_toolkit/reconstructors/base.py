"""Base Class for Reconstructors."""

from numpy.typing import NDArray

import logging
from dataclasses import field
from typing import Any, ClassVar
from snake._meta import MetaDCRegister
from snake.mrd_utils import MRDLoader
from snake.simulation import SimConfig


class MetaReconstructor(MetaDCRegister):
    """MetaClass Reconstructor."""

    dunder_name = "reconstructor"


class BaseReconstructor(metaclass=MetaReconstructor):
    """Represents the interface required to be benchmark-able."""

    __registry__: ClassVar[dict]
    __reconstructor_name__: ClassVar[str]
    __requires__: ClassVar[list[str]]
    log: ClassVar[logging.Logger]

    nufft_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # TODO: Check if all module in requires are available
        pass

    def setup(self, sim_conf: SimConfig) -> None:
        """Set up the reconstructor."""
        self.log.info(f"Setup reconstructor {self.__class__.__name__}")

    def reconstruct(self, data_loader: MRDLoader, sim_conf: SimConfig) -> NDArray:
        """Reconstruct the kspace data to image space."""
        raise NotImplementedError


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
