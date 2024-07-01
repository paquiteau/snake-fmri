"""Base handler module."""

import logging
from .._meta import MetaDCRegister
import numpy as np
from typing import ClassVar, TypeVar, Any
from collections.abc import Callable

from ..simulation import SimConfig
from ..phantom import Phantom

T = TypeVar("T")


class MetaHandler(MetaDCRegister):
    """MetaClass for Handlers."""

    dunder_name = "handler"


class AbstractHandler(metaclass=MetaHandler):
    """Handler Interface.

    An Handler is designed to modify a Simulation data object.

    Handlers can be chained using the ``@`` operator.
    Once created, an handler (and its chain of other registered handler) can be applied
    on a simulation using the `handle` function

    Examples
    --------
    >>> A = Handler()
    >>> B = Handler()
    >>> C = Handler() >> A
    >>> s1 = Simulation()
    >>> C.handle(s1.copy()) == B.handle(A.handle(s1))

    """

    __handler_name__: ClassVar[str]

    @property
    def log(self) -> logging.Logger:
        """Get a logger."""
        return logging.getLogger(f"simulation.handlers.{self.__class__.__name__}")

    def get_static(phantom: Phantom, sim_conf: SimConfig) -> Phantom:
        """Get the static information of the handler."""
        pass

    def get_dynamic(
        phantom: Phantom, sim_conf: SimConfig
    ) -> tuple[T, Callable[[Phantom, T], Phantom]]:
        """Get the dynamic information of the handler."""
        pass
