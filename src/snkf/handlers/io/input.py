"""Handlers for loading data into the simulation."""

import os

from omegaconf import OmegaConf

from snkf.handlers.base import AbstractHandler
from snkf.simulation import SimData

from .utils import load_data


class LoadSimHandler(AbstractHandler):
    """Handler to load a simulation from a file.

    TODO Add support for hdf5 files.

    Parameters
    ----------
    sim_pkl_file
        Filepath to load the data
    """

    __handler_name__ = "load-sim"

    sim_file: str
    dtype: str = "float32"

    def _handle(self, sim: SimData) -> SimData:
        """Load the simulation using pickle."""
        return sim.load_from_file(self.sim_file, dtype=self.dtype)


class LoadDataHandler(AbstractHandler):
    """Load data into fields.

    Parameters
    ----------
    **kwargs
        The fields to load into.  The key is the field name, the value is the path.

    """

    __handler_name__ = "load-data"

    def __init__(self, **kwargs: os.PathLike):
        self.fields = kwargs

    def _handle(self, sim: SimData) -> SimData:
        """Load data to sim."""
        for field, path in self.fields.items():
            self.log.debug(f"Loading {path} into {field}")
            data = load_data(path)
            setattr(sim, field, data)
        return sim


class UpdateFieldHandler(AbstractHandler):
    """Update a field of the simulation with a value.

    Parameters
    ----------
    **kwargs: Mapping[str, Any]
        The field to update and the value to update it with.
        The key is the field name, the value is the value to update it with.
    """

    __handler_name__ = "load-field"

    def __init__(self, **kwargs: None):
        self.field_updates = OmegaConf.to_container(kwargs)

    def _handle(self, sim: SimData) -> SimData:
        """Update the field."""
        for k, v in self.field_updates:
            if isinstance(getattr(sim, k), dict) and isinstance(v, dict):
                setattr(sim, k, getattr(sim, k) | v)
            else:
                setattr(sim, k, v)
        return sim
