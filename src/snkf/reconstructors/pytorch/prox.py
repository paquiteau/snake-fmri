import torch
import numpy as np
import cupy as cp


class WaveletSoftThreshold:
    """Soft thresholding for wavelet coefficicents using pytorch."""

    def __init__(self, thresh: float | torch.Tensor | list[torch.Tensor]):
        self.thresh = thresh
        self.relu = torch.nn.ReLU()

    def op(self, data: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply Soft Thresholding to all coeffs."""
        for d in data[1:]:
            denom = d.abs()
            max_val = self.relu(1.0 - self.thresh / denom)
            d.copy_(max_val * d)
        return data
