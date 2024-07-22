"""Base handler module."""

from .._meta import MetaDCRegister
from typing import ClassVar, TypeVar

from ..simulation import SimConfig
from ..phantom import Phantom, DynamicData, KspaceDynamicData

T = TypeVar("T")


class MetaHandler(MetaDCRegister):
    """MetaClass for Handlers."""

    dunder_name = "handler"


class AbstractHandler(metaclass=MetaHandler):
    """Handler Interface."""

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
