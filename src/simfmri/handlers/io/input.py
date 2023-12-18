"""Handlers for loading data into the simulation."""

import os
from typing import Mapping

from omegaconf import OmegaConf

from simfmri.handlers.base import AbstractHandler
from simfmri.simulation import SimData

from .utils import load_data


class LoadDataHandler(AbstractHandler):
    """Load data into fields.

    Parameters
    ----------
    **kwargs
        The fields to load into.  The key is the field name, the value is the path to the data.

    """

    name = "load-data"

    def __init__(self, **kwargs: Mapping[str, os.PathLike]):
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

    name = "load-field"

    def __init__(self, **kwargs: Mapping[str, os.PathLike]):
        self.field_updates = OmegaConf.to_container(kwargs)

    def _handle(self, sim: SimData) -> SimData:
        """Update the field."""
        for k, v in self.field_updates:
            if isinstance(getattr(sim, k), dict) and isinstance(v, dict):
                setattr(sim, k, getattr(sim, k) | v)
            else:
                setattr(sim, k, v)
        return sim
