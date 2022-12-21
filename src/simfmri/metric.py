"""Metric function to compare to different array. """

import numpy as np


def get_signal_noise(
    test: np.ndarray, ref: np.ndarray, roi: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
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
        noise = abs(ref) - abs(test)
    else:
        signal = ref
        noise = test - ref

    if roi is not None:
        signal = signal[roi]
        noise = noise[roi]

    return signal, noise


def get_snr(test: np.ndarray, ref: np.ndarray, roi: np.ndarray = None) -> float:
    """Compute the overall SNR between data_ref and data_acq"""
    signal, noise = get_signal_noise(test, ref, roi)

    return np.sqrt(np.mean(signal**2) / np.mean(noise**2))


def get_tsnr(
    test: np.ndarray,
    ref: np.ndarray,
    roi: np.ndarray = None,
    tax: int = 0,
) -> np.ndarray:
    """Get the tSNR

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

    return np.sqrt(np.mean(signal**2, axis=tax) / np.mean(noise**2, axis=tax))


def get_ptsnr(
    test: np.ndarray,
    ref: np.ndarray,
    roi: np.ndarray = None,
    tax: int = 0,
) -> float:
    """Get the peak tSNR."""
    return np.max(get_tsnr(test, ref, roi, tax))


def get_snr_axis(test, ref, roi=None, mean_axis=0, peak_axis=None):
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


    Notes
    -----

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
