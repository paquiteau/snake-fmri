"""Proximal Operator reimplemented using pytorch."""

import torch
import numpy as np


class WaveletSoftThreshold:
    """Soft thresholding for wavelet coefficicents using pytorch."""

    def __init__(self, thresh: float | list[torch.Tensor | float]):
        self.thresh = thresh
        self.relu = torch.nn.ReLU()

    def op(
        self, data: list[torch.Tensor], extra_factor: float | torch.Tensor = 1.0
    ) -> list[torch.Tensor]:
        """Apply Soft Thresholding to all coeffs."""
        if isinstance(self.thresh, float):
            self.thresh = [self.thresh] * (len(data) - 1)
        for i, d in enumerate(data[1:]):
            denom = d.abs()
            max_val = self.relu(1.0 - self.thresh[i - 1] * extra_factor / denom)
            d.copy_(max_val * d)
        return data

    def cost(self, data: list[torch.Tensor]) -> float:
        """Compute weighted L1 Norm cost."""
        if isinstance(self.thresh, float):
            self.thresh = [self.thresh] * (len(data) - 1)
        cost = torch.Tensor([0]).to(data[0].device)
        for i, d in enumerate(data):
            cost += torch.sum(self.thresh[i] * d.abs())
        return cost.item()


def _sigma_mad(data: torch.Tensor, centered: bool = True) -> float:
    r"""Return a robust estimation of the standard deviation.

    The standard deviation is computed using the following estimator, based on the
    Median Absolute deviation of the data [#]_
    .. math::
        \hat{\sigma} = \frac{MAD}{\sqrt{2}\textrm{erf}^{-1}(1/2)}

    Parameters
    ----------
    data: numpy.ndarray
        the data on which the standard deviation will be estimated.
    centered: bool, default True.
        If true the median of the is assummed to be 0.

    Returns
    -------
    float:
        The estimation of the standard deviation.

    References
    ----------
    .. [#] https://en.m.wikipedia.org/wiki/Median_absolute_deviation
    """
    if centered:
        return torch.median(torch.abs(data[:])) / 0.6745
    return torch.median(torch.abs(data[:] - torch.median(data[:]))) / 0.6745


def _sure_est(data: torch.Tensor) -> torch.Tensor:
    """Return an estimation of the threshold computed using the SURE method.

    The computation of the estimator is based on the formulation of `cite:donoho1994`
    and the efficient implementation of [#]_

    Parameters
    ----------
    data: numpy.array
        Noisy Data with unit standard deviation.

    Returns
    -------
    float
        Value of the threshold minimizing the SURE estimator.

    References
    ----------
    .. [#] https://pyyawt.readthedocs.io/_modules/pyyawt/denoising.html#ValSUREThresh
    """
    dataf = torch.flatten(data)
    n = np.prod(dataf.size())
    data_sorted, indices = torch.sort(torch.abs(dataf) ** 2)
    idx = torch.arange(n - 1, -1, -1, device=dataf.device)
    tmp = torch.cumsum(data_sorted, 0) + idx * data_sorted

    risk = (n - (2 * torch.arange(n, device=dataf.device)) + tmp) / n
    ibest = torch.argmin(risk)

    return torch.sqrt(data_sorted[ibest])


def _thresh_select(data: torch.Tensor, thresh_est: str) -> torch.Tensor:
    """
    Threshold selection for denoising.

    Implements the methods in `cite:donoho1994`.

    Parameters
    ----------
    data: torch.Tensor
        Noisy data on which a threshold will be estimated. It should only be corrupted
        by a standard gaussian white noise N(0,1).
    thresh_est: str
        threshold estimation method. Available are "sure", "universal", "hybrid-sure".

    Returns
    -------
    float:
        the threshold for the data provided.
    """
    n = np.prod(data.size())
    universal_thr = np.sqrt(2 * np.log(n))

    if thresh_est not in ["sure", "universal", "hybrid-sure"]:
        raise ValueError(
            "Unsupported threshold method."
            "Available are 'sure', 'universal' and 'hybrid-sure'"
        )

    if thresh_est == "sure":
        thr = _sure_est(data)
    elif thresh_est == "universal":
        thr = universal_thr
    elif thresh_est == "hybrid-sure":
        eta = torch.sum(data**2) / n - 1
        if eta.item() < (np.log2(n) ** 1.5) / np.sqrt(n):
            return universal_thr
        test_th = _sure_est(data)
        return min(test_th, universal_thr)
    return thr


def wavelet_noise_estimate(
    wavelet_coeffs: list[torch.Tensor],
    sigma_est: str,
) -> torch.Tensor:
    r"""Return an estimate of the noise standard deviation in each subband.

    Parameters
    ----------
    wavelet_coeffs: numpy.ndarray
        flatten array of wavelet coefficients, typically returned by ``WaveletN.op``
    coeffs_shape:
        list of tuple representing the shape of each subband.
        Typically accessible by WaveletN.coeffs_shape
    sigma_est: str
        Estimation method, available are "band", "scale", and "global"

    Returns
    -------
    numpy.ndarray
        Estimation of the variance for each wavelet subband.

    Notes
    -----
    This methods makes several assumptions:

     - The wavelet coefficients are ordered by scale and the scales are ordered by size.
     - At each scale, the subbands should have the same shape.

    The variance estimation is either performed:

     - On each subband (``sigma_est = "band"``)
     - On each scale, using the detailled HH subband. (``sigma_est = "scale"``)
     - Only with the largest, most detailled HH band (``sigma_est = "global"``)

    See Also
    --------
    _sigma_mad: function estimating the standard deviation.
    """
    sigma_ret = torch.ones(len(wavelet_coeffs), device=wavelet_coeffs[0].device)
    sigma_ret[0] = 0  # approximation coefficient is not computed.
    if sigma_est is None:
        return sigma_ret
    elif sigma_est == "band":
        for i in range(1, len(wavelet_coeffs)):
            sigma_ret[i] = _sigma_mad(wavelet_coeffs[i])
    elif sigma_est == "scale":
        # use the diagonal coefficients subband to estimate the variance of the scale.
        # it assumes that the band of the same scale have the same shape.

        n_subband = 2 ** len(wavelet_coeffs[0].shape) - 1
        for i in range((len(wavelet_coeffs) - 1) // n_subband):
            sigma_ret[1 + i * n_subband : 1 + (i + 1) * n_subband] = _sigma_mad(
                wavelet_coeffs[1 + i * n_subband]
            )
    elif sigma_est == "global":
        # use the last detailled subband.
        sigma_ret *= _sigma_mad(wavelet_coeffs[-1])
    return sigma_ret


def wavelet_threshold_estimate(
    wavelet_coeffs: list[torch.Tensor],
    thresh_range: str = "global",
    sigma_range: str = "global",
    thresh_estimation: str = "hybrid-sure",
) -> torch.Tensor:
    """Estimate wavelet coefficient thresholds.

    Notes that no threshold will be estimate for the coarse scale.

    Parameters
    ----------
    wavelet_coeffs: numpy.ndarray
        flatten array of wavelet coefficient, typically returned by ``WaveletN.op``
    coeffs_shape: list
        List of tuple representing the shape of each subbands.
        Typically accessible by WaveletN.coeffs_shape
    thresh_range: str. default "global"
        Defines on which data range to estimate thresholds.
        Either "band", "scale", or "global"
    sigma_range: str, default "global"
        Defines on which data range to estimate thresholds.
        Either "band", "scale", or "global"
    thresh_estimation: str, default "hybrid-sure"
        Name of the threshold estimation method.
        Available are "sure", "hybrid-sure", "universal"

    Returns
    -------
    numpy.ndarray
        array of threshold for each wavelet coefficient.
    """
    weights = torch.ones(len(wavelet_coeffs), device=wavelet_coeffs[0].device)
    weights[0] = 0

    n_subband = 2 ** len(wavelet_coeffs[0].shape) - 1
    # Estimate the noise std on the specific range.

    sigma_bands = wavelet_noise_estimate(wavelet_coeffs, sigma_range)

    # compute the threshold on each specific range.

    if thresh_range == "global":
        weights[:] = sigma_bands[-1] * _thresh_select(
            wavelet_coeffs[-1],
            thresh_estimation,
        )
    elif thresh_range == "band":
        for i in range((len(wavelet_coeffs) - 1) // n_subband):
            band_slicer = slice(1 + i * n_subband, 1 + (i + 1) * n_subband)
            all_band_coeffs = torch.cat(
                [
                    t.flatten() / s
                    for t, s in zip(
                        wavelet_coeffs[band_slicer],
                        sigma_bands[band_slicer],
                    )
                ]
            )

            t = sigma_bands[band_slicer] * _thresh_select(
                all_band_coeffs, thresh_estimation
            )
            weights[band_slicer] = t

    elif thresh_range == "scale":
        for i in range((len(wavelet_coeffs) - 1) // n_subband):
            band_slicer = slice(1 + i * n_subband, 1 + (i + 1) * n_subband)
            detail_slicer = i * n_subband

            t = sigma_bands[detail_slicer] * _thresh_select(
                wavelet_coeffs[detail_slicer] / sigma_bands[detail_slicer],
                thresh_estimation,
            )
            weights[band_slicer] = t
    return weights


class AutoWaveletSoftThreshold(WaveletSoftThreshold):
    """
    Automatic estimation of Wavelet Soft Threshold parameters using SURE.

    Parameters
    ----------
    coeffs_shape: list
        list of coefficient shape.

    update_period: int

    """

    def __init__(
        self,
        update_period: int = 0,
        sigma_range: str = "global",
        thresh_range: str = "global",
        threshold_estimation: str = "sure",
        threshold_scaler: float = 1.0,
    ):
        self._n_op_calls = 0
        self._update_period = update_period

        if thresh_range not in ["bands", "scale", "global"]:
            raise ValueError("Unsupported threshold range.")
        if sigma_range not in ["bands", "scale", "global"]:
            raise ValueError("Unsupported sigma estimation method.")
        if threshold_estimation not in ["sure", "hybrid-sure", "universal", "bayes"]:
            raise ValueError("Unsupported threshold estimation method.")

        self._sigma_range = sigma_range
        self._thresh_range = thresh_range
        self._thresh_estimation = threshold_estimation
        self._thresh_scale = threshold_scaler

        super().__init__(thresh=0)

    def _auto_thresh(self, input_data: list[torch.Tensor]) -> list[torch.Tensor]:
        """Compute the best weights for the input_data.

        Parameters
        ----------
        input_data: numpy.ndarray
            Array of sparse coefficient.

        See Also
        --------
        wavelet_threshold_estimate
        """
        weights = wavelet_threshold_estimate(
            input_data,
            thresh_range=self._thresh_range,
            sigma_range=self._sigma_range,
            thresh_estimation=self._thresh_estimation,
        )
        if callable(self._thresh_scale):
            weights = self._thresh_scale(weights, self._n_op_calls)
        else:
            weights = [w * self._thresh_scale for w in weights]
        return weights

    def op(
        self, input_data: list[torch.Tensor], extra_factor: float = 1.0
    ) -> list[torch.Tensor]:
        """Operator.

        This method returns the input data thresholded by the weights.
        The weights are computed using the universal threshold rule.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor (default is ``1.0``)

        Returns
        -------
        numpy.ndarray
            Thresholded data

        """
        if self._update_period == 0 and self._n_op_calls == 0:
            self.thresh = self._auto_thresh(input_data)
        if self._update_period != 0 and self._n_op_calls % self._update_period == 0:
            self.thresh = self._auto_thresh(input_data)

        self._n_op_calls += 1
        return super().op(input_data, extra_factor=extra_factor)
