"""Pytorch based reconstructors for simfmri data.

Note that we are not using any Deep Learning here, but rather the Pytorch framework
for its streamlined handling of tensors and GPU support.
"""
from __future__ import annotations
from typing import Literal

import numpy as np
from mrinufft import get_operator

from ..simulation import SimData
from .base import BaseReconstructor

TORCH_AVAILABLE = True
try:
    import ptwt
    import torch
except ImportError:
    TORCH_AVAILABLE = False

CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False


class TorchWaveletTransform:
    """Wavelet transform using pytorch."""

    wavedec3_keys = ["aad", "ada", "add", "daa", "dad", "dda", "ddd"]

    def __init__(
        self,
        shape: tuple[int, ...],
        wavelet: str,
        level: int,
        mode: str,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not available.")
        self.wavelet = wavelet
        self.level = level
        self.shape = shape
        self.mode = mode

    def op(self, data: torch.Tensor) -> list[torch.Tensor]:
        """Apply the wavelet decomposition."""
        if len(self.shape) == 2:
            if torch.is_complex(data):
                # 2D Complex
                data_ = torch.view_as_real(data)
                coeffs_ = ptwt.wavedec2(
                    data_, self.wavelet, level=self.level, mode=self.mode, axes=(-3, -2)
                )
                # flatten list of tuple of tensors to a list of tensors
                coeffs = [torch.view_as_complex(coeffs_[0].contiguous())] + [
                    torch.view_as_complex(cc.contiguous())
                    for c in coeffs_[1:]
                    for cc in c
                ]

                return coeffs
            # 2D Real
            coeffs_ = ptwt.wavedec2(
                data, self.wavelet, level=self.level, mode=self.mode, axes=(-2, -1)
            )
            return [coeffs_[0]] + [cc for c in coeffs_[1:] for cc in c]

        if torch.is_complex(data):
            # 3D Complex
            data_ = torch.view_as_real(data)
            coeffs_ = ptwt.wavedec3(
                data_,
                self.wavelet,
                level=self.level,
                mode=self.mode,
                axes=(-4, -3, -2),
            )
            # flatten list of tuple of tensors to a list of tensors
            coeffs = [torch.view_as_complex(coeffs_[0].contiguous())] + [
                torch.view_as_complex(cc.contiguous())
                for c in coeffs_[1:].values()
                for cc in c
            ]

            return coeffs
        # 3D Real
        coeffs_ = ptwt.wavedec3(
            data, self.wavelet, level=self.level, mode=self.mode, axes=(-3, -2, -1)
        )
        return [coeffs_[0]] + [cc for c in coeffs_[1:].values() for cc in c]

    def adj_op(self, coeffs: list[torch.Tensor]) -> torch.Tensor:
        """Apply the wavelet recomposition."""
        if len(self.shape) == 2:
            if torch.is_complex(coeffs[0]):
                ## 2D Complex ##
                # list of tensor to list of tuple of tensor
                coeffs = [torch.view_as_real(coeffs[0])] + [
                    tuple(torch.view_as_real(coeffs[i + k]) for k in range(3))
                    for i in range(1, len(coeffs) - 2, 3)
                ]
                data = ptwt.waverec2(coeffs, wavelet=self.wavelet, axes=(-3, -2))
                return torch.view_as_complex(data.contiguous())
            ## 2D Real ##
            coeffs_ = [coeffs[0]] + [
                tuple(coeffs[i + k] for k in range(3))
                for i in range(1, len(coeffs) - 2, 3)
            ]
            data = ptwt.waverec2(coeffs_, wavelet=self.wavelet, axes=(-2, -1))
            return data

        if torch.is_complex(coeffs[0]):
            ## 3D Complex ##
            # list of tensor to list of tuple of tensor
            coeffs = [torch.view_as_real(coeffs[0])] + [
                {
                    v: torch.view_as_real(coeffs[i + k])
                    for k, v in enumerate(self.wavedec3_keys)
                }
                for i in range(1, len(coeffs) - 6, 7)
            ]
            data = ptwt.waverec3(coeffs, wavelet=self.wavelet, axes=(-4, -3, -2))
            return torch.view_as_complex(data.contiguous())
        ## 3D Real ##
        coeffs_ = [coeffs[0]] + [
            {v: coeffs[i + k] for k, v in enumerate(self.wavedec3_keys)}
            for i in range(1, len(coeffs) - 6, 7)
        ]
        data = ptwt.waverec3(coeffs_, wavelet=self.wavelet, axes=(-3, -2, -1))
        return data


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


class SequentialReconstructor(BaseReconstructor):
    """Sequential reconstruction using pytorch and cufinufft."""

    name = "sequential-torch"

    def __init__(
        self,
        max_iter_per_frame: int = 15,
        optimizer: str = "pogm",
        wavelet: str = "sym4",
        threshold: float | Literal["sure"] = "sure",
        **kwargs,
    ):
        super().__init__(nufft_backend, nufft_kwargs)

        self.max_iter_per_frame = max_iter_per_frame
        self.optimizer = optimizer
        self.wavelet_name = wavelet
        self.threshold = threshold

    def setup(self, sim: SimData) -> None:
        """Set up the reconstructor."""
        self.wavelet = TorchWaveletTransform(
            sim.shape,
            wavelet=self.wavelet_name,
            level=3,
            mode="zero",
        )
        self.prox = WaveletSoftThreshold(self.threshold)

    def reconstruct(self, sim: SimData) -> np.ndarray:
        """Reconstruct data."""
        n_frames = len(sim.kspace_data)
        init_frame = np.zeros(sim.shape, dtype=np.complex64)
        final = np.zeros((n_frames, *sim.shape), dtype=np.complex64)
        smaps = cp.array(sim.smaps) if sim.smaps is not None else None

        # TODO: Parallel ?
        for i, (ksp_data, ksp_mask) in enumerate(zip(sim.kspace_data, sim.kspace_mask)):
            final[i] = self._reconstruct(
                ksp_data,
                ksp_mask,
                prev_frame=init_frame,
                smaps=smaps,
                shape=sim.shape,
                n_coils=sim.n_coils,
            )

    def _reconstruct(
        self,
        kspace_data: np.ndarray,
        kspace_mask: np.ndarray,
        previous_frame: np.ndarray,
        shape: tuple,
        smaps: cp.ndarray | None = None,
        n_coils: int = 1,
    ) -> np.ndarray:
        """Reconstruct a single frame of data using FISTA."""
        nufft = get_operator("cufinufft")(
            kspace_mask,
            n_coils=n_coils,
            density="cell-count",
            upsampfac=1.25,
            gpu_kerevalmeth=2,
        )
        L = nufft.get_lipschitz_cst(max_iter=10, upsampfac=1.25, gpu_kerevalmeth=2)
        eta = 1 / L
        xk = torch.Tensor(*shape, dtype=torch.complex64)
        tk = 1
        kspace_data = cp.array(kspace_data)
        for i in range(self.max_iter_per_frame):
            # Fista loop
            x_tmp = self.wavelet.adj_op(
                self.prox.op(
                    self.wavelet.op(
                        xk - eta * self.nufft.data_consistency(xk, kspace_data)
                    )
                )
            )
            tkk = (1 + np.sqrt(1 + 4 * tk**2)) // 2
            xkk = x_tmp + ((tk - 1) / tkk) * (x_tmp - xk)
            xk.copy_(xkk)
            tk = tkk

        return xkk.to("cpu").numpy()
