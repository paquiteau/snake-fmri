"""Reconstructor interface.

Reconstructor implement a `reconstruct` method which takes a simulation object as input
and returns a reconstructed fMRI array.
"""

from __future__ import annotations
from dataclasses import field
from typing import Any
import logging
import numpy as np

from modopt.opt.linear import Identity

from .base import BaseReconstructor, SpaceFourierProto
from snkf.simulation import SimData

logger = logging.getLogger(__name__)


def get_fourier_operator(
    sim: SimData,
    backend_name: str | None = None,
    density: str = "cell_count",
    **kwargs: Any,
) -> SpaceFourierProto:
    """Return a Fourier operator for the given simulation."""
    kwargs = kwargs.copy() if kwargs is not None else {}

    from fmri.operators.fourier import (
        CartesianSpaceFourier,
        LazySpaceFourier,
        PooledgpuNUFFTSpaceFourier,
        RepeatOperator,
    )
    from mrinufft import get_operator

    if backend_name is None:
        backend_name = ""

    if backend_name == "gpunufft-cpu":
        return PooledgpuNUFFTSpaceFourier(
            sim.kspace_mask,
            sim.shape,
            n_frames=len(sim.kspace_data),
            n_coils=sim.n_coils,
            smaps=sim.smaps,
            density=density,
            pool_size=5,
            **kwargs,
        )

    smaps = sim.smaps
    if backend_name in ["cufinufft", "stacked-cufinufft"]:
        import cupy as cp

        if smaps is not None:
            smaps = cp.array(smaps)
        kwargs["smaps_cached"] = True
    if sim.extra_infos["traj_params"]["constant"] is True and backend_name != "":
        logger.debug("using a duplicated operator.")
        return RepeatOperator(
            [
                get_operator(
                    backend_name=backend_name,
                    samples=sim.kspace_mask[0],
                    shape=sim.shape,
                    n_coils=sim.n_coils,
                    smaps=smaps,
                    density=density,
                    squeeze_dims=True,
                    **kwargs,
                )
            ]
            * len(sim.kspace_data)
        )

    if "stacked" in backend_name:
        kwargs["z_index"] = "auto"
    if "nufft" in backend_name or "stacked" in backend_name:

        return LazySpaceFourier(
            backend=backend_name,
            samples=sim.kspace_mask,
            shape=sim.shape,
            n_frames=len(sim.kspace_data),
            n_coils=sim.n_coils,
            smaps=smaps,
            density=density,
            squeeze_dims=True,
            **kwargs,
        )

    return CartesianSpaceFourier(
        shape=sim.kspace_mask.shape[1:],
        mask=sim.kspace_mask,
        n_frames=len(sim.kspace_data),
        n_coils=sim.n_coils,
        smaps=sim.smaps,
        **kwargs,
    )


class ZeroFilledReconstructor(BaseReconstructor):
    """Reconstruction using zero-filled (ifft) method."""

    __reconstructor_name__ = "adjoint"

    def __post_init__(self):
        super().__post_init__()
        self.fourier_op = None

    def setup(self, sim: SimData) -> None:
        """Set up the reconstructor."""
        self.fourier_op: SpaceFourierProto = get_fourier_operator(
            sim, **self.nufft_kwargs
        )

    def reconstruct(self, sim: SimData) -> np.ndarray:
        """Reconstruct with Zero filled method."""
        if self.fourier_op is None:
            self.setup(sim)
        return self.fourier_op.adj_op(sim.kspace_data)


class SequentialReconstructor(BaseReconstructor):
    """Use a sequential Reconstruction.

    Parameters
    ----------
    max_iter_frame
        Number of iteration to allow per frame.
    optimizer
        Optimizer name, available are pogm and fista.
    threshold
        Threshold value for the wavelet regularisation.
    """

    __reconstructor_name__ = "sequential"

    max_iter_per_frame: int = 15
    optimizer: str = "pogm"
    wavelet: str = "sym8"
    threshold: float | str = "sure"
    nufft_kwargs: dict[str, Any] = field(default_factory=dict)
    compute_backend: str = "cupy"
    restart_strategy: str = "warm"

    def setup(self, sim: SimData) -> None:
        """Set up the reconstructor."""
        from modopt.opt.linear.wavelet import WaveletTransform
        from fmri.operators.weighted import AutoWeightedSparseThreshold
        from modopt.opt.proximity import SparseThreshold

        self.fourier_op = get_fourier_operator(
            sim,
            **self.nufft_kwargs,
        )

        space_linear_op = WaveletTransform(
            self.wavelet,
            shape=sim.shape,
            level=3,
            mode="zero",
            compute_backend=self.compute_backend,
        )

        _ = space_linear_op.op(np.zeros(sim.shape, dtype=sim.kspace_data.dtype))

        if self.threshold == "sure":
            space_prox_op = AutoWeightedSparseThreshold(
                space_linear_op.coeffs_shape,
                linear=None,
                threshold_estimation="hybrid-sure",
                threshold_scaler=0.6,
            )
        else:
            self.threshold = float(self.threshold)
            space_prox_op = SparseThreshold(linear=Identity(), weights=self.threshold)
        from fmri.reconstructors.frame_based import (
            SequentialReconstructor,
            DoubleSequentialReconstructor,
        )

        if self.restart_strategy == "warm-double":
            rec_klass = DoubleSequentialReconstructor
        else:
            rec_klass = SequentialReconstructor

        self.reconstructor: SequentialReconstructor = rec_klass(
            self.fourier_op,
            space_linear_op,
            space_prox_op,
            optimizer=self.optimizer,
        )

    def reconstruct(self, sim: SimData, fourier_op: None = None) -> np.ndarray:
        """Reconstruct with Sequential."""
        self.setup(sim)

        seq_rec = self.reconstructor.reconstruct(
            sim.kspace_data,
            max_iter_per_frame=self.max_iter_per_frame,
            compute_backend=self.compute_backend,
            restart_strategy=self.restart_strategy,
        )
        return seq_rec


class LowRankPlusSparseReconstructor(BaseReconstructor):
    """Low Rank + Sparse Benchmark reconstructor.

    Parameters
    ----------
    lr_thresh
        regularisation parameter for low rank prior
    sparse_thresh
        regularisation parameter for sparse prior
    max_iter
        maximal number of interation.
    """

    __reconstructor_name__ = "lr_f"

    nufft_kwargs: dict[str, Any] = field(default_factory=dict)
    lambda_l: float = 0.1
    lambda_s: float | str = 0.1
    algorithm: str = "otazo"
    max_iter: int = 20

    def __post_init__(self):
        super().__post_init__()
        self.time_linear_op = None
        self.time_prox_op = None
        self.space_prox_op = None
        self.fourier_op = None

    def __str__(self):
        if isinstance(self.lambda_s, float):
            return f"LRS-{self.lambda_l:.2e}-{self.lambda_s:.2e}"
        return f"LRS-{self.lambda_l:.2e}-{self.lambda_s}"

    def setup(self, sim: SimData) -> None:
        """Set up the reconstructor."""
        from fmri.reconstructors.time_aware import LowRankPlusSparseReconstructor
        from fmri.operators.utils import sure_est, sigma_mad
        from fmri.operators.proximity import InTransformSparseThreshold
        from fmri.operators.time_op import TimeFourier
        from fmri.operators.svt import FlattenSVT

        if self.fourier_op is None:
            self.fourier_op = get_fourier_operator(
                sim,
                **self.nufft_kwargs,
            )

        logger.debug("Space Fourier operator initialized")
        if self.time_linear_op is None:
            self.time_linear_op = TimeFourier(time_axis=0)

        logger.debug("Time Fourier operator initialized")
        if self.lambda_s == "sure":
            adj_data = self.fourier_op.adj_op(sim.kspace_data)
            sure_thresh = np.zeros(np.prod(adj_data.shape[1:]))
            tf = self.time_linear_op.op(adj_data).reshape(len(adj_data), -1)
            for i in range(len(sure_thresh)):
                sure_thresh[i] = sure_est(tf[:, i]) * sigma_mad(tf[:, i])

            self.lambda_s = np.median(sure_thresh) / 2
            logger.info(f"SURE threshold: {self.lambda_s:.4e}")

        if self.time_prox_op is None and self.time_linear_op is not None:
            self.time_prox_op = InTransformSparseThreshold(
                self.time_linear_op, self.lambda_s, thresh_type="soft"
            )

        logger.debug("Prox Time  operator initialized")
        if self.space_prox_op is None:
            self.space_prox_op = FlattenSVT(
                self.lambda_l, initial_rank=10, thresh_type="soft-rel"
            )
        logger.debug("Prox Space operator initialized")

        self.reconstructor: LowRankPlusSparseReconstructor = (
            LowRankPlusSparseReconstructor(
                self.fourier_op,
                space_prox_op=self.space_prox_op,
                time_prox_op=self.time_prox_op,
                cost="auto",
            )
        )
        logger.debug("Reconstructor initialized")

    def reconstruct(
        self, sim: SimData, fourier_op: SpaceFourierProto | None = None
    ) -> np.ndarray:
        """Reconstruct using LowRank+Sparse Method."""
        if fourier_op is not None:
            self.fourier_op = fourier_op
        self.setup(sim)
        M, L, S, costs = self.reconstructor.reconstruct(
            sim.kspace_data,
            grad_step=0.5,
            max_iter=self.max_iter,
            optimizer=self.algorithm,
        )
        return M


class LowRankPlusTVReconstructor(LowRankPlusSparseReconstructor):
    """Low Rank + TV."""

    ...


class LowRankPlusWaveletReconstructor(LowRankPlusSparseReconstructor):
    """Low Rank + Wavelet."""

    ...
    # SURE threhsold estimated properly with the AutoWeighted Sparse Threhsold.


class ZeroFilledOptimalThreshReconstructor(ZeroFilledReconstructor):
    """
    Reconstructor using a simple adjoint and a denoiser.

    Parameters
    ----------
    patch_shape
    patch_overlap
    recombination

    TODO Add support for different denoising methods.

    """

    __reconstructor_name__ = "adjoint+denoised"

    def __init__(
        self,
        patch_shape: int,
        patch_overlap: int,
        recombination: str = "weighted",
        nufft_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(nufft_kwargs)
        self.patch_shape = patch_shape
        self.patch_overlap = patch_overlap
        self.recombination = recombination

    def __str__(self):
        return (
            f"Denoiser_{self.patch_shape}-{self.patch_overlap}-{self.recombination[0]}"
        )

    def reconstruct(self, sim: SimData) -> np.ndarray:
        """Reconstruct using a simple adjoint and denoiser."""
        from denoiser.denoise import optimal_thresholding

        data_zerofilled = super().reconstruct(sim)

        llr_denoise = optimal_thresholding(
            np.moveaxis(data_zerofilled, 0, -1),
            patch_shape=self.patch_shape,
            patch_overlap=self.patch_overlap,
            recombination=self.recombination,
        )
        return np.moveaxis(llr_denoise[0], -1, 0)
