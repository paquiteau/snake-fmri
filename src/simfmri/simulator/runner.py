"""Runners for the simulator."""

from abc import ABC, abstractmethod


from .simulation import Simulation
from .handler import (
    AbstractHandler,
    PhantomGeneratorHandler,
)


class AbstractSimulatorRunner:
    """A Simulation Runner provides a simple interface for creating simulation data.

    In practise, it defines the recipie to get a specific  Simulation


    Attributes
    ----------
    handlers: list of Handler
        Represent the chain of handler responsability.
        The first handler of the list should
    current_simulation: Simulation
        Current simulation state.
    """
