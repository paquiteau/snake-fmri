"""Output Handlers for saving data from the simulation."""

import os
from pathlib import Path
from typing import Any, Mapping

import pickle
from omegaconf import OmegaConf

from snkf.simulation import SimData
from snkf.handlers.base import AbstractHandler, requires_field

from .utils import save_data


class SaveSimHandler(AbstractHandler):
    """Save data to a file."""

    __handler_name__ = "save-sim"

    def __init__(self, filename: os.PathLike):
        self.filename = Path(filename)

    def _handle(self, sim: SimData) -> SimData:
        """Save the simulation to a file."""
        pickle.dump(sim, open(self.filename, "wb"))
        return sim


class SaveFieldHandler(AbstractHandler):
    """Save specific fields to files."""

    __handler_name__ = "save-fields"

    def __init__(self, **kwargs: os.PathLike[Any]):
        self.fields = kwargs

    def _handle(self, sim: SimData) -> SimData:
        for field, path in self.fields.items():
            data = getattr(sim, field)
            self.log.debug(f"Saving {field} to {self.fields[field]}")
            save_data(data, path)

        return sim


@requires_field("kspace_data")
@requires_field("kspace_mask")
class ISMRMRDHandler(AbstractHandler):
    """Save the kspace data and trajectory to an ISMRMRD file."""

    def __init__(self, filename: os.PathLike, header: Mapping[str, Any]):
        self.filename = Path(filename)
        self.header = OmegaConf.to_container(header)

    def _handle(self, sim: SimData) -> SimData:

        raise NotImplementedError
        # Complete the minimal header with user input
        # Create The ISMRMRD dataset

        # dset = ismrmrd.Dataset(self.filename, self.headers)
