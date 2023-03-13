"""Noise Module.

This module declares the various noise models availables.
"""
from __future__ import annotations
from .base import AbstractHandler
from ..simulation import SimulationData

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

    def _callback_fun(self, old_sim: SimulationData, new_sim: SimulationData) -> None:
        # TODO compute the SNR and print it.
        pass

    def _handle(self, sim: SimulationData) -> SimulationData:
        if self._snr == 0:
            return sim
        else:
            # SNR is defined as average(brain signal) / noise_std
            noise_std = np.mean(abs(sim.data_ref > 0)) / self._snr
            self._add_noise(sim, noise_std)

        sim.extra_infos["input_snr"] = self._snr
        return sim

    def _add_noise(self, sim: SimulationData) -> None:
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

    def _add_noise(self, sim: SimulationData, noise_std: float) -> None:
        if np.iscomplexobj(sim.data_ref):
            noise_std /= np.sqrt(2)
        noise = noise_std * self._rng.standard_normal(
            sim.data_ref.shape, dtype=abs(sim.data_ref[:][0]).dtype
        )
        noise = noise.astype(sim.data_ref.dtype)

        if sim.data_ref.dtype in [np.complex128, np.complex64]:
            noise += (
                1j
                * noise_std
                * self._rng.standard_normal(
                    sim.data_ref.shape,
                    dtype=abs(sim.data_ref[:][0]).dtype,
                )
            )

        sim.data_acq = sim.data_ref + noise


class RicianNoiseHandler(NoiseHandler):
    """Add rician noise to the data."""

    def _add_noise(self, sim: SimulationData, noise_std: float) -> None:
        if np.any(np.iscomplex(sim)):
            raise ValueError(
                "The Rice distribution is only applicable to real-valued data."
            )
        sim.data_acq = sps.rice(
            sim.data_ref,
            scale=noise_std,
        )


class KspaceNoiseHandler(NoiseHandler):
    """Add gaussian in the kspace."""

    def _add_noise(self, sim: SimulationData, noise_std: float) -> None:
        if sim.kspace_data is None:
            raise ValueError("kspace data not initialized.")

        # Complex Value, so the std is spread.
        noise_std /= np.sqrt(2)

        kspace_noise = np.complex64(
            self._rng.standard_normal(sim.kspace_data.shape, dtype="float32")
        )
        kspace_noise += 1j * self._rng.standard_normal(
            sim.kspace_data.shape, dtype="float32"
        )
        kspace_noise *= noise_std

        sim.kspace_data += kspace_noise * sim.kspace_mask
