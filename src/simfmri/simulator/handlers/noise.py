"""Noise Module

This module declares the various noise models availables.
"""
from __future__ import annotations
from .base import AbstractHandler
from ..simulation import Simulation

from simfmri.utils import validate_rng, RngType

import numpy as np
import scipy.stats as sps


class NoiseHandler(AbstractHandler):
    """Add noise to the data.

    Parameters
    ----------
    snr
        The target SNR
        The  SNR  is defined as max(signal) / std(noise)
    rng
        Random Generator, optional,  int or numpy random state
    verbose
        verbose flag.
        If True, a callback function is setup to return the computed snr.
    """

    def __init__(self, verbose: bool = True, rng: RngType = None, snr: float = 0):
        super().__init__()
        self._verbose = verbose
        self._rng = validate_rng(rng)
        if self._verbose:
            self.add_callback(self._callback_fun)

        self._snr = snr

    def _callback_fun(self, old_sim, new_sim):
        if self._verbose:
            # compute the SNR and print it.
            print("actual_snr:")

    def _handle(self, sim: Simulation):
        if self._snr == 0:
            return sim
        else:
            sigma_noise = np.max(abs(sim.data_ref)) / self._snr
            self._add_noise(sim, sigma_noise)

        sim.extra_infos["input_snr"] = self._snr

    def _add_noise(self, sim: Simulation):
        """add noise to the simulation.

        This should only update the attribute data_acq  of a Simulation object.

        Parameters
        ----------
        sim
            Simulation data object
        """
        raise NotImplementedError


class GaussianNoiseHandler(NoiseHandler):
    """Add gaussian Noise to the data."""

    def _add_noise(self, sim: Simulation, sigma_noise: float):

        noise = sigma_noise * self._rng.standard_normal(
            sim.data_ref.shape, dtype=sim.data_ref.dtype
        )

        if np.iscomplex(sim[:][0]):
            noise += 1j * sigma_noise * self._rng.standard_normal(sim.data_ref.shape)

        sim.data_acq = sim.data_ref + noise


class RicianNoiseHandler(NoiseHandler):
    """Add rician noise to the data."""

    def _add_noise(self, sim: Simulation, sigma_noise: float):
        if np.any(np.iscomplex(sim)):
            raise ValueError(
                "The Rice distribution is only applicable to real-valued data."
            )
        sim.data_acq = sps.rice(
            sim.data_ref,
            scale=sigma_noise,
        )


class KspaceNoiseHandler(NoiseHandler):
    """Add gaussian in the kspace."""

    def _add_noise(self, sim: Simulation, sigma_noise: float):
        if sim.kspace_data is None:
            raise ValueError("kspace data not initialized.")

        kspace_noise = self._rng.standard_normal(
            sim.kspace_data.shape, dtype=sim.kspace_data.dtype
        )
        kspace_noise += 1j * self._rng.standard_normal(sim.kspace_data.shape)
        kspace_noise *= sigma_noise

        sim.kspace_data += kspace_noise * sim.kspace_mak
