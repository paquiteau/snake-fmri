"""Handler to load existing simulation data."""

import pickle
from .base import AbstractHandler
from ..siulation import SimulationData


class LoadDataHandler(AbstractHandler):
    """Handler to load a simulation from a file.

    TODO Add support for hdf5 files.

    Parameters
    ----------
    sim_pkl_file
        Filepath to load the data
    """

    def __init__(self, sim_pkl_file: str):

        self.sim_pkl_file = self.sim_pkl_file

    def _handle(self, sim: SimulationData) -> SimulationData:
        """Load the simulation using pickle."""
        return pickle.load(self.sim_pkl_file)


class SaveDataHandler(AbstractHandler):
    """Handler to save a simulation to a file.

    The current backend is pickle.

    TODO Add support for hdf5 files.

    Parameters
    ----------
    sim_pkl_file
        Filepath to load the data
    """

    def __init__(self, sim_pkl_file: str):
        self.sim_pkl_file = self.sim_pkl_file

    def _handle(self, sim: SimulationData) -> SimulationData:
        """Save the simulation using pickle."""
        with open(self.sim_pkl_file, "w") as f:

            pickle.dump(sim, f)
        return sim
