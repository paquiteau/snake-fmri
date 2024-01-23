"""Metric function to compare two different arrays."""

import numpy as np
from numpy.typing import NDArray


def get_signal_noise(
    test: NDArray, ref: NDArray, roi: NDArray | None = None
) -> tuple[NDArray, NDArray]:
    """Get the actual signal and noise, in real-valued setting.

    If the data is complex, only the magnitude are compared.

    Parameters
    ----------
    test: test data
    ref: reference data
    roi: (optional) region of interest.

    Returns
    -------
    tuple:
        np.ndarray: signal data
        np.ndarray: noise data (test - ref )

    """
    if np.any(np.iscomplex(ref)):
        signal = abs(ref)
        noise: NDArray = abs(test) - abs(signal)
    else:
        signal = ref
        noise = test - ref

    if roi is not None:
        signal = signal[roi]
        noise = noise[roi]

    return signal, noise


def get_snr(test: NDArray, ref: NDArray, roi: NDArray | None = None) -> float:
    """
    Compute the overall SNR between test (signal+noise) and ref(signal only).

    Parameters
    ----------
    test: test data
    ref: reference data
    roi: (optional) region of interest.

    """
    signal, noise = get_signal_noise(test, ref, roi)

    return np.sqrt(np.mean(abs(signal) ** 2) / np.mean(abs(noise) ** 2))


def get_tsnr(
    test: NDArray,
    ref: NDArray,
    roi: NDArray | None = None,
    tax: int = 0,
) -> np.ndarray:
    """Get the tSNR.

    Parameters
    ----------
    test: test array
    ref: reference array (noise-free)
    tax: axis reference for time, default is `0`

    Returns
    -------
    ptsnr: peak tsnr of the volume.
    """
    signal, noise = get_signal_noise(test, ref, roi)

    return np.sqrt(
        np.mean(abs(signal) ** 2, axis=tax) / np.mean(abs(noise) ** 2, axis=tax)
    )


def get_ptsnr(
    test: NDArray,
    ref: NDArray,
    roi: NDArray | None = None,
    tax: int = 0,
) -> float:
    """Get the peak tSNR."""
    return np.max(get_tsnr(test, ref, roi, tax))


def get_snr_axis(
    test: NDArray,
    ref: NDArray,
    roi: NDArray | None = None,
    mean_axis: int = 0,
    peak_axis: int | None = None,
) -> np.ndarray:
    """Get the snr, computed over the mean axis.

    Parameters
    ----------
    test: test data
    ref: reference data
    roi: region of interest mask
    mean_axis: int or tuple of int

    Returns
    -------
    np.ndarray: the computed snr.
    """
    if mean_axis == peak_axis:
        raise ValueError("mean axis and peak axis should be different.")

    signal, noise = get_signal_noise(test, ref, roi)
    # fmt: off
    snr = (np.mean(signal**2, axis=mean_axis, keepdims=True)
           / np.mean(noise**2, axis=mean_axis, keepdims=True))
    # fmt: on
    if peak_axis is not None:
        snr = np.max(snr, axis=peak_axis)
    return np.squeeze(snr)
