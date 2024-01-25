"""Pytorch based reconstructors for simfmri data.

Note that we are not using any Deep Learning here, but rather the Pytorch framework
for its streamlined handling of tensors and GPU support.
"""
from __future__ import annotations
import logging
from typing import Literal
from tqdm.auto import tqdm

import numpy as np
from mrinufft import get_operator

from ...simulation import SimData
from ..base import BaseReconstructor
from .wavelet import TorchWaveletTransform
from .prox import WaveletSoftThreshold

TORCH_AVAILABLE = True
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False

CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False

logger = logging.getLogger("TorchReconstructor")


class TorchSequentialReconstructor(BaseReconstructor):
    """Sequential reconstruction using pytorch and cufinufft."""

    name = "sequential-torch"

    def __init__(
        self,
        max_iter_per_frame: int = 15,
        optimizer: str = "pogm",
        wavelet: str = "sym4",
        threshold: float | Literal["sure"] = "sure",
        nufft_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__()

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
        for i in tqdm(range(n_frames), position=0):
            final[i] = self._reconstruct(
                sim.kspace_data[i],
                sim.kspace_mask[i],
                previous_frame=init_frame,
                smaps=smaps,
                shape=sim.shape,
                n_coils=sim.n_coils,
            )
        return final

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
            shape=shape,
            n_coils=n_coils,
            smaps=smaps,
            smaps_cached=True,
            density="cell_count",
        )
        logger.debug("Estimating Lipschitz constant...")
        L = nufft.get_lipschitz_cst(max_iter=10)
        eta = 1 / L
        logger.debug(f"step_size: {eta}")
        xk = torch.empty(1, 1, *shape, dtype=torch.complex64, device="cuda")
        xk.copy_(torch.from_numpy(previous_frame))
        tk = 1
        kspace_data = torch.from_numpy(kspace_data).to("cuda")
        for i in tqdm(range(self.max_iter_per_frame), position=1, leave=False):
            # Fista loop
            grad = nufft.data_consistency(xk, kspace_data)
            x_tmp = xk - eta * grad
            x_tmp = self.wavelet.adj_op(self.prox.op(self.wavelet.op(x_tmp)))
            tkk = (1 + np.sqrt(1 + 4 * tk**2)) // 2
            xkk = x_tmp + ((tk - 1) / tkk) * (x_tmp - xk)
            xk.copy_(xkk)
            tk = tkk

        return xkk.to("cpu").numpy().squeeze()
