"""Noise Module.

This module declares the various noise models availables.
"""
from __future__ import annotations
from .base import AbstractHandler
from ..simulation import SimData, LazySimArray

from simfmri.utils import validate_rng

import numpy as np
import scipy.stats as sps


def _lazy_add_noise(
    data: np.ndarray,
    noise_std: float,
    root_seed: int,
    frame_idx: int = None,
) -> np.ndarray:
    """Add noise to data."""
    rng = np.random.default_rng([frame_idx, root_seed])
    if data.dtype in [np.complex128, np.complex64]:
        noise_std /= np.sqrt(2)
    noise = noise_std * rng.standard_normal(data.shape, dtype=abs(data[:][0]).dtype)
    noise = noise.astype(data.dtype)

    if data.dtype in [np.complex128, np.complex64]:
        noise += (
            1j
            * noise_std
            * rng.standard_normal(
                data.shape,
                dtype=abs(data[:][0]).dtype,
            )
        )
    return data + noise


class BaseNoiseHandler(AbstractHandler):
    """Add noise to the data.

    Parameters
    ----------
    snr
        The target SNR
        The  SNR  is defined as max(signal) / std(noise)
    verbose
        verbose flag.
        If True, a callback function is setup to return the computed snr.
    """

    def __init__(self, snr: float = 0):
        super().__init__()
        self._snr = snr

    def _handle(self, sim: SimData) -> SimData:
        if self._snr == 0:
            return sim
        else:
            # SNR is defined as average(brain signal) / noise_std
            noise_std = np.mean(abs(sim.static_vol[sim.static_vol > 0])) / self._snr
        if sim.lazy:
            self._add_noise_lazy(sim, sim.rng, noise_std)
        else:
            self._add_noise(sim, sim.rng, noise_std)

        sim.extra_infos["input_snr"] = self._snr
        return sim

    def _add_noise(self, sim: SimData, noise_std: float) -> None:
        """Add noise to the simulation.

        This should only update the attribute data_acq  of a Simulation object
        and use the ``_rng`` attribute to generate the noise.

        Parameters
        ----------
        sim
            Simulation data object
        """
        raise NotImplementedError

    def _add_noise_lazy(self, sim: SimData, noise_std: float) -> None:
        """Lazily add noise to the simulation.

        This should only update the attribute data_acq  of a Simulation object
        and use the ``_rng`` attribute to generate the noise.

        Parameters
        ----------
        sim
            Simulation data object
        """
        raise NotImplementedError


class GaussianNoiseHandler(BaseNoiseHandler):
    """Add gaussian Noise to the data."""

    name = "noise-gaussian"

    def _add_noise(self, sim: SimData, rng_seed: int, noise_std: float) -> None:
        rng = validate_rng(rng_seed)
        if np.iscomplexobj(sim.data_ref):
            noise_std /= np.sqrt(2)
        noise = noise_std * rng.standard_normal(
            sim.data_ref.shape, dtype=abs(sim.data_ref[:][0]).dtype
        )
        noise = noise.astype(sim.data_ref.dtype)

        if sim.data_ref.dtype in [np.complex128, np.complex64]:
            noise += (
                1j
                * noise_std
                * rng.standard_normal(
                    sim.data_ref.shape,
                    dtype=abs(sim.data_ref[:][0]).dtype,
                )
            )
        if sim.data_acq is None:
            sim.data_acq = sim.data_ref.copy()
        sim.data_acq = sim.data_acq + noise

    def _add_noise_lazy(self, sim: SimData, rng_seed: int, noise_std: float) -> None:
        if sim.data_acq is None:
            sim.data_acq = LazySimArray(sim.data_ref, len(sim.data_ref))

        sim.data_acq.apply(_lazy_add_noise, noise_std, rng_seed)


class RicianNoiseHandler(BaseNoiseHandler):
    """Add rician noise to the data."""

    name = "noise-rician"

    def _add_noise(self, sim: SimData, noise_std: float) -> None:
        if np.any(np.iscomplex(sim)):
            raise ValueError(
                "The Rice distribution is only applicable to real-valued data."
            )
        sim.data_acq = sps.rice(
            sim.data_ref,
            scale=noise_std,
        )


class KspaceNoiseHandler(BaseNoiseHandler):
    """Add gaussian in the kspace."""

    name = "noise-kspace"

    def _handle(self, sim: SimData) -> SimData:
        if self._snr == 0:
            return sim
        else:
            # SNR is defined as average(brain signal) / noise_std
            noise_std = np.mean(abs(sim.static_vol > 0)) / self._snr
        self._add_noise(sim, sim.rng, noise_std)

        sim.extra_infos["input_snr"] = self._snr
        return sim

    def _add_noise(self, sim: SimData, rng_seed: int, noise_std: float) -> None:
        rng = validate_rng(rng_seed)
        if sim.kspace_data is None:
            raise ValueError("kspace data not initialized.")

        # Complex Value, so the std is spread.
        noise_std /= np.sqrt(2)
        for kf in range(len(sim.kspace_data)):
            kspace_noise = np.complex64(
                rng.standard_normal(sim.kspace_data.shape[1:], dtype="float32")
            )
            kspace_noise += 1j * rng.standard_normal(
                sim.kspace_data.shape[1:], dtype="float32"
            )
            kspace_noise *= noise_std

            if sim.extra_infos["traj_name"] == "vds":
                if sim.n_coils > 1:
                    sim.kspace_data[kf] += kspace_noise * sim.kspace_mask[kf, None, ...]
                else:
                    sim.kspace_data[kf] += kspace_noise * sim.kspace_mask[kf]
            else:
                sim.kspace_data[kf] += kspace_noise
