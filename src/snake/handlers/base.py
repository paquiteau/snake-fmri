"""Base handler module."""

from __future__ import annotations
import yaml
import dataclasses
from collections import UserList

from .._meta import MetaDCRegister
from typing import ClassVar, TypeVar, Any

from ..simulation import SimConfig
from ..phantom import Phantom, DynamicData, KspaceDynamicData

T = TypeVar("T")


class MetaHandler(MetaDCRegister):
    """MetaClass for Handlers."""

    dunder_name = "handler"


class AbstractHandler(metaclass=MetaHandler):
    """Handler Interface."""

    __registry__: ClassVar[dict[str, type[AbstractHandler]]]
    __handler_name__: ClassVar[str]

    def get_static(self, phantom: Phantom, sim_conf: SimConfig) -> Phantom:
        """Get the static information of the handler."""
        return phantom

    def get_dynamic(self, phantom: Phantom, sim_conf: SimConfig) -> DynamicData | None:
        """Get the dynamic information of the handler."""
        return None

    def get_dynamic_kspace(self, sim_conf: SimConfig) -> KspaceDynamicData | None:
        """Get the dynamic kspace information of the handler."""
        return None

    def to_yaml(self) -> str:
        """Show the yaml config associated with the handler."""
        return yaml.dump(dataclasses.asdict(self))  # type: ignore


class HandlerList(UserList):
    """Represent a Chain of Handler, that needs to be apply to a simulation."""

    def __init__(self, *args: AbstractHandler):
        self.data = list(args)

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> HandlerList:
        """Create a HandlerList from a configuration."""
        self = cls()
        for handler_name, handler_cfg in cfg.items():
            handler = get_handler(handler_name)
            self.append(handler(**handler_cfg))
        return self

    def to_yaml(self, filename: str | None = None) -> str:
        """Serialize the handlerList as a yaml string."""
        if filename:
            with open(filename, "w") as f:
                yaml.dump(self.serialize(), f)
            return filename
        else:
            return yaml.dump(self.serialize())

    def serialize(self) -> list[dict[str, Any]]:
        """Serialize the handlerList as a list of dictionnary."""
        # handlers are dataclasses.
        return [dataclasses.asdict(h) for h in self.data]


# short alias
H = AbstractHandler.__registry__
handler = H


def list_handlers() -> list[str]:
    """List all available handlers."""
    return list(H.keys())


def get_handler(name: str) -> type[AbstractHandler]:
    """Get a handler from its name."""
    return H[name]
