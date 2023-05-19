"""Example of scenarios for the simulator."""

from typing import Mapping

import numpy as np
from omegaconf import DictConfig, OmegaConf

from simfmri.simulator.handlers import (
    AbstractHandler,
)
from simfmri.simulator.simulation import SimulationData, SimulationParams


class SimulationDataFactory:
    """Simulation Factory.

    The Simulation data is generated throught the `simulate` function.

    Parameters
    ----------
    sim_params
        Essential parameters for a simulation
    handlers
        list of handler to apply to get the fully generated simulation data.

    Notes
    -----
    This is best used with hydra configuration files. see `conf/simulations/`
    """

    def __init__(
        self,
        sim_params: SimulationParams,
        handlers: list[AbstractHandler] | Mapping[str, AbstractHandler],
        checkpoints: bool = False,
    ):
        self.checkpoints = checkpoints
        self.sim_params = sim_params
        if isinstance(handlers, DictConfig):
            self.handlers = list(handlers.values())
        else:
            self.handlers = handlers

    def simulate(self) -> SimulationData:
        """Build the simulation data."""
        sim = SimulationData.from_params(self.sim_params)

        for idx, handler in enumerate(self.handlers):
            sim = handler.handle(sim)
            if self.checkpoints:
                np.save(f"{idx}-{handler}.npy", sim.data_ref)
        return sim
