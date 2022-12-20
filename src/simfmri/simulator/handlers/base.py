"""Base Handler Interface."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Callable

from .simulation import Simulation


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
    >>> C = Handler() @ A
    >>> s1 = Simulation()
    >>> C.handle(s1.copy()) == B.handle(A.handle(s1))

    """

    _next: AbstractHandler = None

    def __init__(self):
        self._callbacks = None

    def __matmul__(self, obj: AbstractHandler):
        if isinstance(obj, AbstractHandler):
            return self.set_next(obj)
        else:
            raise TypeError

    def _run_callback(self, old_sim, new_sim):
        """Run the different callbacks."""

        if isinstance(self._callbacks, list):
            for callback_fun in self._callbacks:
                callback_fun(old_sim, new_sim)
        else:
            try:
                self._callbacks(old_sim, new_sim)
            except TypeError as e:
                raise RuntimeError("callback function not callable.") from e

    @property
    def callbacks(self):
        return self._callback

    def add_callback(self, call: Callable):
        if not callable(call):
            raise TypeError("Callback attribute should be callable with two argument.")
        if not isinstance(self._callbacks, list):
            self._callback = [self._callbacks]

        self._callbacks.append(call)

    def remove_callback(self, idx):
        """Remove callback according to its position."""
        self._callback.pop(idx)

    def set_next(self, handler: AbstractHandler) -> AbstractHandler:
        self._next = handler
        return handler

    def handle(self, sim: Simulation) -> Simulation:
        if self._callback is not None:
            old_sim = copy.deepcopy(sim)
            new_sim = self._handle(sim)
            self._run_callbacks(old_sim, new_sim)
        else:
            new_sim = self._handle(sim)
        if self._next_handler:
            return self._next.handle(new_sim)
        else:
            return new_sim

    @abstractmethod
    def _handle(self, sim: Simulation) -> None:
        pass
