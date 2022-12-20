from .base import AbstractHandler
from ..simulation import Simulation

from simfmri.utils import validate_rng, RngType

import numpy as np
import scipy.stats as sps


class NoiseHandler(AbstractHandler):
    """Add noise to the data

    Parameters
    ----------
    snr: The target SNR
    rng: int or numpy random state
    verbose: verbose flag.
        If True, a callback function is setup to return the computed snr.

    """

    def __init__(self, verbose=True, rng: RngType = None, snr: float = 0):
        super().__init__()
        self._verbose = verbose
        self._rng = validate_rng(rng)
        if self._verbose:
            self.add_callback(self._callback_fun)

    def _callback_fun(self, old_sim, new_sim):
        if self._verbose:
            # compute the SNR and print it.
            print("actual_snr:")

    def _handle(self, sim: Simulation):
        if self._snr == 0:
            return sim
        else:
            self._add_noise(sim)

    def _add_noise(self, sim):
        raise NotImplementedError


class GaussianNoiseHandler(NoiseHandler):
    """Add gaussian Noise to the data."""

    def __init__(self, snr=0, rng=None):
        self._snr = snr


class RicianNoiseHandler(NoiseHandler):
    """
    Add rician noise to the data.

    """
