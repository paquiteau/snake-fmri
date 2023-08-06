"""Base Handler Interface."""
from __future__ import annotations
import time
import copy
import logging
from abc import ABC, abstractmethod
from typing import Callable, Any

from ..simulation import SimDataType


CallbackType = Callable[[SimDataType, SimDataType], Any]


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

    _callbacks: list[CallbackType]

    def __init__(self) -> None:
        self._callbacks = []
        self.next = None
        self.prev = None

    def __rshift__(self, obj: AbstractHandler) -> AbstractHandler:
        """Chain the handler with the righhandside, and return the righhandside.

        Example
        -------
        >>> a, b = Handler(), Handler()
        >>> a >> b is a.set_next(b)
        True
        >>> a >> b is b
        True
        """
        if isinstance(obj, AbstractHandler):
            self.next = obj
            obj.prev = self
            return obj
        else:
            raise TypeError

    def __call__(self, sim: SimDataType) -> SimDataType:
        """Short-hand for handle operation."""
        return self.handle(sim)

    def __str__(self) -> str:
        return self.__class__.__name__

    def _run_callbacks(self, old_sim: SimDataType, new_sim: SimDataType) -> None:
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

    def handle(self, sim: SimDataType) -> SimDataType:
        """Handle a specific action done on the simulation, and move to the next one."""
        if self.prev is not None:
            sim = self.prev.handle(sim)
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
    def _handle(self, sim: SimDataType) -> SimDataType:
        pass
