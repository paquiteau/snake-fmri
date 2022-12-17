from __future__ import annotations
from typing import Callable
from abc import ABC, abstractmethod
import copy
import numpy as np

from .simulation import Simulation


class AbstractHandler(ABC):
    """Base simulation class"""

    _next: AbstractHandler = None

    def __init__(self):
        self._callback = None

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, call: Callable):
        if callable(call):
            self._callback = call
        else:
            raise ValueError

    def set_next(self, handler: AbstractHandler) -> AbstractHandler:
        self._next = handler
        return handler

    @abstractmethod
    def handle(self, sim: Simulation) -> Simulation:
        if self._next_handler:
            if self.callback is not None:
                old_sim = copy.deepcopy(sim)
                new_sim = self._next.handle(sim)
                self.callback(old_sim, new_sim)
                return new_sim
        else:
            return sim


class PhantomGeneratorHandler(AbstractHandler):
    """Handler to create the base phantom.

    phantom generation should be the first step of the simulation.
    Moreover, it only accept 3D shape.
    """

    def handle(self, sim: Simulation):
        return super().handle(sim)


class SlicerHandler(AbstractHandler):
    """Get only one slice of the data.

    This update the shape of all the array in the simulation.
    """

    def handle(self, sim: Simulation, axis: int = -1, cut=None):
        return super().handle(sim)


class MotionHandler(AbstractHandler):
    """Add Motion to the data."""

    def handle(self, sim: Simulation, motion_course: np.ndarray):
        return super.handle(sim)


class NoiseHandler(AbstractHandler):
    def handle(self, sim: Simulation, noise: np.ndarray):
        return super.handle(sim)


class GaussianNoiseHandler(AbstractHandler):
    def handle(self, sim: Simulation, noise: np.ndarray):
        return super.handle(sim)


class RicianNoiseHandler(AbstractHandler):
    def handle(self, sim: Simulation, noise: np.ndarray):
        return super.handle(sim)
