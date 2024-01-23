"""Handler to load existing simulation data."""

from .base import AbstractHandler
from ..simulation import SimData


class LoadDataHandler(AbstractHandler):
    """Handler to load a simulation from a file.

    TODO Add support for hdf5 files.

    Parameters
    ----------
    sim_pkl_file
        Filepath to load the data
    """

    def __init__(self, sim_file: str, dtype: str = "float32"):
        super().__init__()
        self.sim_file = sim_file
        self.dtype = dtype

    def _handle(self, sim: SimData) -> SimData:
        """Load the simulation using pickle."""
        return sim.load_from_file(self.sim_file, dtype=self.dtype)


class SaveDataHandler(AbstractHandler):
    """Handler to save a simulation to a file.

    The current backend is pickle.

    TODO Add support for hdf5 files.

    Parameters
    ----------
    sim_pkl_file
        Filepath to load the data
    """

    def __init__(self, sim_file: str):
        super().__init__()
        self.sim_file = sim_file

    def _handle(self, sim: SimData) -> SimData:
        """Save the simulation using pickle."""
        sim.save(self.sim_file)
        return sim
