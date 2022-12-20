"""Runners for the simulator."""

from abc import ABC, abstractmethod


from .simulation import Simulation
from .handler import (
    AbstractHandler,
    PhantomGeneratorHandler,
)


class AbstractSimulator(ABC):
    """A Simulation Runner provides a simple interface for creating simulation data.

    Parameters
    ----------

    Attributes
    ----------
    handlers: list of Handler
        Represent the chain of handler responsability.
        The first handler of the list should
    """

    def __init__(self, init_sim):
        self._current_sim = None
        self.glob_handler = None
        self._init_simulation()

    @property
    def current_sim(self):
        """Current simulation."""
        return self._current_sim

    @current_sim.setter
    def current_sim(self, simul: Simulation):
        if isinstance(simul, Simulation):
            self._current_sim = simul
        else:
            raise ValueError("simul arg should be a simulation.")

    def add_callback(self, handler: AbstractHandler) -> None:
        """Add a callback to the handler"""

        def _callback(old: Simulation, new: Simulation):
            self.sim_list.append(old)

        handler.callback = _callback

    @abstractmethod
    def _init_simulation(self):
        pass


class StaticSimulationRunner:
    """A simple simulation generator where the data is a repeated acquition."""

    def __init__(self, x, y, z=1, t=1):
        if z == 1:
            shape = x, y, max(x, y)
        else:
            shape = x, y, z

        super().__init__(Simulation(shape=shape, n_frames=t))

    def _init_simulation(self):
        print("static Simulation")

        self.handler = PhantomGeneratorHandler()
