"""Base Handler Interface."""
from __future__ import annotations
from typing import Callable, Union
from abc import ABC, abstractmethod
import copy

from .simulation import Simulation


class AbstractHandler(ABC):
    """Base simulation class"""

    _next: AbstractHandler = None

    def __init__(self):
        self._callback = None

    def __matmul__(self, obj: Union(Simulation, AbstractHandler)):
        if isinstance(obj, AbstractHandler):
            return self.set_next(obj)
        elif isinstance(obj, Simulation):
            return self.handle(obj)
        else:
            raise TypeError

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, call: Callable):
        if callable(call):
            self._callback = call
        else:
            raise ValueError("Callback attribute should be callable with two argument.")

    def set_next(self, handler: AbstractHandler) -> AbstractHandler:
        self._next = handler
        return handler

    def handle(self, sim: Simulation) -> Simulation:
        if self._callback is not None:
            old_sim = copy.deepcopy(sim)
            new_sim = self._handle(sim)
            self._callback(old_sim, new_sim)
        else:
            new_sim = self._handle(sim)
        if self._next_handler:
            return self._next.handle(new_sim)
        else:
            return new_sim

    @abstractmethod
    def _handle(self, sim: Simulation) -> None:
        pass
