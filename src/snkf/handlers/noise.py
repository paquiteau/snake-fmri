"""Noise Module.

This module declares the various noise models availables.
"""

from __future__ import annotations
from .base import AbstractHandler, requires_field
from ..simulation import SimData, LazySimArray

from snkf.base import validate_rng, RngType, real_type

import numpy as np
from numpy.typing import NDArray
import scipy.stats as sps


def _lazy_add_noise(
    data: NDArray[np.complexfloating] | NDArray[np.floating],
    noise_std: float,
    root_seed: int,
    frame_idx: int = 0,
) -> np.ndarray:
    """Add noise to data."""
    rng = np.random.default_rng([frame_idx, root_seed])
    if data.dtype in [np.complex128, np.complex64]:
        noise_std /= np.sqrt(2)
    noise = (
        noise_std * rng.standard_normal(data.shape, dtype=real_type(data.dtype))
    ).astype(data.dtype)

    if data.dtype in [np.complex128, np.complex64]:
        noise = noise * 1j
        noise += noise_std * rng.standard_normal(
            data.shape,
            dtype=abs(data[:][0]).dtype,
        )
    return data + noise


require_data_acq = requires_field("data_acq", lambda sim: sim.data_ref.copy())


@require_data_acq
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

    snr: float

    def _handle(self, sim: SimData) -> SimData:
        if self.snr == 0:
            return sim
        else:
            # SNR is defined as average(brain signal) / noise_std
            noise_std = np.mean(abs(sim.static_vol[sim.static_vol > 0])) / self.snr
        if isinstance(sim.data_acq, LazySimArray):
            self._add_noise_lazy(sim.data_acq, sim.rng, noise_std)
        else:
            self._add_noise(sim, sim.rng, noise_std)

        sim.extra_infos["inputsnr"] = self.snr
        return sim

    def _add_noise(self, sim: SimData, rng_seed: int, noise_std: float) -> None:
        """Add noise to the simulation.

        This should only update the attribute data_acq  of a Simulation object
        and use the ``_rng`` attribute to generate the noise.

        Parameters
        ----------
        sim
            Simulation data object
        """
        raise NotImplementedError

    def _add_noise_lazy(
        self, lazy_arr: LazySimArray, rng_seed: int, noise_std: float
    ) -> None:
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

    __handler_name__ = "noise-gaussian"
    snr: int | float

    def _add_noise(self, sim: SimData, rng_seed: RngType, noise_std: float) -> None:
        rng = validate_rng(rng_seed)
        if np.iscomplexobj(sim.data_ref):
            noise_std /= np.sqrt(2)
        noise = (
            noise_std
            * rng.standard_normal(
                sim.data_acq.shape, dtype=real_type(sim.data_acq.dtype)
            )
        ).astype(sim.data_acq.dtype)

        if sim.data_acq.dtype in [np.complex128, np.complex64]:
            noise = noise * 1j
            noise += noise_std * rng.standard_normal(
                sim.data_ref.shape,
                dtype=real_type(sim.data_ref.dtype),
            )
        sim.data_acq += noise
        self.log.debug(f"{sim.data_acq}, {sim.data_ref}")

    def _add_noise_lazy(
        self, lazy_arr: LazySimArray, rng_seed: int, noise_std: float
    ) -> None:
        lazy_arr.apply(_lazy_add_noise, noise_std, rng_seed)


class RicianNoiseHandler(BaseNoiseHandler):
    """Add rician noise to the data."""

    __handler_name__ = "noise-rician"
    snr: int | float

    def _add_noise(self, sim: SimData, rng_seed: int, noise_std: float) -> None:
        if np.any(np.iscomplex(sim)):
            raise ValueError(
                "The Rice distribution is only applicable to real-valued data."
            )
        sim.data_acq = sps.rice(
            sim.data_ref,
            scale=noise_std,
        )


@requires_field("kspace_mask")
@requires_field("kspace_data")
class KspaceNoiseHandler(BaseNoiseHandler):
    """Add gaussian in the kspace."""

    __handler_name__ = "noise-kspace"
    snr: int | float

    def _handle(self, sim: SimData) -> SimData:
        if self.snr == 0:
            return sim
        else:
            # SNR is defined as average(brain signal) / noise_std
            noise_std = np.mean(abs(sim.static_vol > 0)) / self.snr
        self._add_noise(sim, sim.rng, noise_std)

        sim.extra_infos["inputsnr"] = self.snr
        return sim

    def _add_noise(self, sim: SimData, rng_seed: int, noise_std: float) -> None:
        rng = validate_rng(rng_seed)

        # Complex Value, so the std is spread.
        noise_std /= np.sqrt(2)
        for kf in range(len(sim.kspace_data)):
            kspace_noise = rng.standard_normal(
                sim.kspace_data.shape[1:], dtype="float32"
            ).astype("complex64")
            kspace_noise *= 1j
            kspace_noise += rng.standard_normal(
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


@require_data_acq
class ScannerDriftHandler(AbstractHandler):
    """Add Scanner Drift to the data.

    Parameters
    ----------
    drift_model : {'polynomial', 'cosine', None},
        string that specifies the desired drift model

    frame_times : array of shape(n_scans),
        list of values representing the desired TRs

    order : int, optional,
        order of the drift model (in case it is polynomial)

    high_pass : float, optional,
        high-pass frequency in case of a cosine model (in Hz)

    See Also
    --------
    nilearn.glm.first_level.design_matrix._make_drift
    """

    drift_model: str = "polynomial"
    order: int = 1
    high_pass: float | None = None
    drift_intensities: float | np.ndarray = 0.01

    def __post_init__(
        self,
        drift_model: str = "polynomial",
        order: int = 1,
        high_pass: float | None = None,
        drift_intensities: float | np.ndarray = 0.01,
    ):
        if not isinstance(self.drift_intensities, np.ndarray):
            if isinstance(self.drift_intensities, (int, float)):
                self.drift_intensities = np.array([drift_intensities] * order)
            else:
                self.drift_intensities = np.array([drift_intensities])

    def _handle(self, sim: SimData) -> SimData:
        # Nilearn does the heavy lifting for us
        from nilearn.glm.first_level.design_matrix import _make_drift

        if self.drift_model is None:
            return sim
        drift_matrix = _make_drift(
            self.drift_model,
            frame_times=np.linspace(0, sim.sim_time, sim.n_frames),
            order=self.order,
            high_pass=self.high_pass,
        )
        drift_matrix = drift_matrix[:, :-1]  # remove the intercept column

        drift_intensity = np.linspace(1, 1 + self.drift_intensities, sim.n_frames)

        timeseries = drift_intensity @ drift_matrix

        if isinstance(sim.data_acq, LazySimArray):
            raise NotImplementedError(
                "lazy is not compatible with scanner drift (for now)"
            )
        else:
            sim.data_acq[sim.static_vol > 0] *= timeseries

        return sim
