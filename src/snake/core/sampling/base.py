"""Sampling pattern generations."""

from __future__ import annotations
import logging
from typing import ClassVar
from numpy.typing import NDArray

from snake._meta import MetaDCRegister
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
    __engine__: ClassVar[str]
    __registry__: ClassVar[dict[str, type[BaseSampler]]]
    constant: bool = True

    @property
    def log(self) -> logging.Logger:
        """Get a logger."""
        return logging.getLogger(f"simulation.samplers.{self.__class__.__name__}")

    def _single_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate a single frame."""
        raise NotImplementedError

    def get_next_frame(self, sim_conf: SimConfig) -> NDArray:
        """Generate the next frame."""
        if self.constant:
            if not hasattr(self, "_frame"):
                self._frame = self._single_frame(sim_conf)
            return self._frame

        return self._single_shot(sim_conf)

    def _single_shot(self, sim_conf: SimConfig) -> NDArray:
        """Generate a single shot."""
        raise NotImplementedError

    def add_all_acq_mrd(self, dataset: mrd.Dataset, sim_conf: SimConfig) -> mrd.Dataset:
        """Export the Sampling pattern to file."""
        raise NotImplementedError
