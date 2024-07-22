"""Sampling pattern generations."""

import logging
from typing import ClassVar
from numpy.typing import NDArray

from .._meta import MetaDCRegister
from ..simulation import SimConfig

import ismrmrd as mrd


class MetaSampler(MetaDCRegister):
    """MetaClass for Samplers."""

    dunder_name = "sampler"


class BaseSampler(metaclass=MetaSampler):
    """Sampler Interface.

    A Sampler is designed to generate a sampling pattern.

    Examples
    --------
    >>> S = Sampler()
    >>> S.generate()
    """

    __sampler_name__: ClassVar[str]
    engine: str
    in_out: bool = True
    obs_time_ms: int = 25

    @property
    def log(self) -> logging.Logger:
        """Get a logger."""
        return logging.getLogger(f"simulation.samplers.{self.__class__.__name__}")

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate a single frame."""
        raise NotImplementedError

    def _single_shot(self, sim_conf: SimConfig) -> NDArray:
        """Generate a single shot."""
        raise NotImplementedError

    def add_all_acq_mrd(self, dataset: mrd.Dataset, sim_conf: SimConfig) -> mrd.Dataset:
        """Export the Sampling pattern to file."""
        raise NotImplementedError
