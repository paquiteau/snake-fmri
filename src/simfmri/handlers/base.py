"""Base Handler Interface."""
from __future__ import annotations
import time
import copy
import logging
from abc import ABC, abstractmethod
from typing import Callable, Any, Iterable, Mapping

from ..simulation import SimData


CallbackType = Callable[[SimData, SimData], Any]


AVAILABLE_HANDLERS: Mapping[str, AbstractHandler] = {}


def handler(
    name: str, *args: Iterable[Any], **kwargs: Mapping[str, Any]
) -> AbstractHandler | type(AbstractHandler):
    """Create a handler from its name."""
    if args or kwargs:
        return AVAILABLE_HANDLERS[name](*args, **kwargs)
    else:
        return AVAILABLE_HANDLERS[name]


# short alias
H = handler


def list_handlers() -> list[str]:
    """List all available handlers."""
    return list(AVAILABLE_HANDLERS.keys())


def get_handler(name: str) -> type(AbstractHandler):
    """Get a handler from its name."""
    return AVAILABLE_HANDLERS[name]


class AbstractHandler(ABC):
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

    def __init__(self) -> None:
        self._callbacks = []

    def __init_subclass__(cls):
        if getattr(cls, "name", None) is not None:
            AVAILABLE_HANDLERS[cls.name] = cls

    def __rshift__(self, other: AbstractHandler | HandlerChain):
        """Perform self >> other."""
        if isinstance(other, AbstractHandler):
            return HandlerChain(self, other)
        else:
            return NotImplemented

    def __lshift__(self, other: AbstractHandler | HandlerChain):
        """Perform self << other."""
        if isinstance(other, AbstractHandler):
            return HandlerChain(other, self)
        return NotImplemented

    def __call__(self, sim: SimData) -> SimData:
        """Short-hand for handle operation."""
        return self.handle(sim)

    def __str__(self) -> str:
        return self.__class__.__name__

    def _run_callbacks(self, old_sim: SimData, new_sim: SimData) -> None:
        """Run the different callbacks.

        Parameters
        ----------
        old_sim
            The simulation object before handling it
        new_sim
            The simulation obkect after handling it
        """
        if isinstance(self._callbacks, list):
            for callback_fun in self._callbacks:
                callback_fun(old_sim, new_sim)
        else:
            try:
                self._callbacks(old_sim, new_sim)
            except TypeError as e:
                raise RuntimeError("callback function not callable.") from e

    @property
    def callbacks(self) -> list[CallbackType]:
        """Return the list of callbacks run after the handling."""
        return self._callbacks

    def add_callback(self, call: CallbackType) -> None:
        """Add a callback to the callback list.

        Parameters
        ----------
        call
            The callback to add
        """
        if not callable(call):
            raise TypeError("Callback attribute should be callable with two argument.")
        if not isinstance(self._callbacks, list):
            self._callbacks = [self._callbacks]

        self._callbacks.append(call)

    def remove_callback(self, idx: int) -> CallbackType:
        """Remove callback according to its position.

        Parameters
        ----------
        idx
            the index of the callback to remove

        Returns
        -------
        callable
            the removed callback.
        """
        return self._callbacks.pop(idx)

    def handle(self, sim: SimData) -> SimData:
        """Handle a specific action done on the simulation, and move to the next one."""
        if self._callbacks:
            old_sim = copy.deepcopy(sim)

        self.log.debug("start handling")
        tic = time.perf_counter()
        new_sim = self._handle(sim)
        toc = time.perf_counter()
        self.log.debug(f"end handling: {toc-tic:.2f}s")

        if self._callbacks:
            self._run_callbacks(old_sim, new_sim)
        return new_sim

    @property
    def log(self) -> logging.Logger:
        """Log the current action."""
        return logging.getLogger(f"simulation.{self.__class__.__name__}")

    @abstractmethod
    def _handle(self, sim: SimData) -> SimData:
        pass


class HandlerChain:
    """Represent a Chain of Handler, that needs to be apply to a simulation."""

    def __init__(self, *args: list[AbstractHandler]):
        self._handlers = list(args)

    def __lshift__(self, other: AbstractHandler | HandlerChain | SimData):
        """
        Perform self << other.

        If other is a handler: add it to the beginning of the chain.
        If other is a handler chain: Create a new chain with other chain before self.
        Else: raise NotImplementedError
        If other is a simulation: Apply the chain of handler from left to right
        """
        if isinstance(other, AbstractHandler):
            self._handlers.insert(0, other)
            return self
        elif isinstance(other, HandlerChain):
            return HandlerChain(*other._handlers, *self._handlers)
        elif isinstance(other, SimData):
            return self.__call__(other)
        return NotImplemented

    def __rlshift__(self, other: AbstractHandler):
        """
        Perform other << self.

        If other is handler: create new chain with handler at the end
        """
        if isinstance(other, AbstractHandler):
            self._handlers.append(other)
            return self
        return NotImplemented

    def __rshift__(self, other: AbstractHandler | HandlerChain):
        """
        Perform self >> other.

        If other is a handler, add other to the end of the chain.
        If other is a handler chain: Create a new chain with self before other
        Else: raise NotImplementedError
        """
        if isinstance(other, AbstractHandler):
            self._handlers.append(other)
            return self
        elif isinstance(other, HandlerChain):
            return HandlerChain(*self._handlers, *other._handlers)
        return NotImplemented

    def __rrshift__(self, other: AbstractHandler | HandlerChain | SimData):
        """
        Perform other >> self.

        If other is a simulation: run the chain
        else: raise Value Error
        """
        if isinstance(other, AbstractHandler):
            self._handlers.insert(0, other)
            return self
        elif isinstance(other, HandlerChain):
            return HandlerChain(*other._handlers, self._handlers)
        elif isinstance(other, SimData):
            return self.__call__(other)
        return NotImplemented

    def __call__(self, sim: SimData):
        """Apply the handler chain to the simulation.

        We first check if there is a cycling dependency.
        """
        id_list = []
        for h in self._handlers:
            if hid := id(h) not in id_list:
                id_list.append(hid)
            else:
                raise RuntimeError("Cycling list of operator detected.")

        old_sim = sim
        for h in self._handlers:
            new_sim = h.handle(old_sim)
            old_sim = new_sim

        return new_sim

    def add_callback_to_all(self, callback: CallbackType) -> None:
        """Add the same callback to all the handlers."""
        if not callable(callback):
            raise ValueError("Callback should be callable with two arguments.")

        for h in self._handlers:
            h.add_callback(callback)

    def __repr__(self):
        """Represent a simulation."""
        ret = ""
        for h in self._handlers:
            ret += f"{h.name}({id(h)}) >> "
        ret = ret[:-3]
        return ret
