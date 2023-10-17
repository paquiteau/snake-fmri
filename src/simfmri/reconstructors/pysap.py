"""Reconstructor interface.

Reconstructor implement a `reconstruct` method which takes a simulation object as input
and returns a reconstructed fMRI array.
"""
from __future__ import annotations
from typing import Literal
import logging
import warnings
import numpy as np


from fmri.operators.fourier import FFT_Sense, RepeatOperator
from fmri.operators.fourier import CartesianSpaceFourier, SpaceFourierBase
from modopt.opt.linear import LinearParent
from modopt.opt.proximity import ProximityParent
from mrinufft.operators import get_operator
from mrinufft.operators.stacked import traj3d2stacked
from mrinufft.trajectories.density import voronoi

from .base import BaseReconstructor
from simfmri.simulation import SimData

logger = logging.getLogger(__name__)


def _get_stacked_operator(backend: str, sim: SimData) -> RepeatOperator:
    nufft_backend = backend.split("-")[1]
    frame_ops = []
    Ns = sim.extra_infos["traj_params"]["n_samples"]
    logger.info(f"{nufft_backend}, {Ns} points in shots")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Samples .*", category=UserWarning, module="mrinufft"
        )
        if nufft_backend == "finufft":
            density = voronoi(sim.kspace_mask[0][:Ns, :2])
        kwargs = dict(
            shape=sim.shape,
            density=density,
            smaps=sim.smaps,
            n_coils=sim.n_coils,
            squeeze_dims=True,
            backend_name=backend,
        )
        if backend != "stacked-cufinufft":
            kwargs["backend_name"] = "stacked"
            kwargs["backend"] = nufft_backend

        for kf in range(sim.kspace_mask.shape[0]):
            traj2d, z_indexes = traj3d2stacked(sim.kspace_mask[kf], sim.shape[-1])
            frame_ops.append(
                get_operator(**kwargs, z_index=np.int32(z_indexes), samples=traj2d)
            )
    return RepeatOperator(frame_ops)


def get_fourier_operator(
    sim: SimData, repeat: bool = False, **kwargs: None
) -> RepeatOperator | CartesianSpaceFourier:
    """Return a Fourier operator for the given simulation."""
    kwargs = kwargs.copy() if kwargs is not None else {}

    density = True
    backend = sim.extra_infos.get("operator", None)
    logger.info(f"fourier backend is {backend}")
    if "stacked-" in backend:
        return _get_stacked_operator()

    if "finufft" in backend:
        kwargs["squeeze_dims"] = True
    kwargs["density"] = density

    def _get_op(i: int = 0) -> SpaceFourierBase:
        return get_operator(backend)(
            sim.kspace_mask[i],
            shape=sim.shape,
            n_coils=sim.n_coils,
            smaps=sim.smaps,
            **kwargs,
        )

    if sim.extra_infos["traj_constant"]:
        return RepeatOperator([_get_op(0)] * len(sim.kspace_data))
    return RepeatOperator([_get_op(i) for i in range(len(sim.kspace_data))])

    if repeat:
        return RepeatOperator(
            [
                FFT_Sense(
                    shape=sim.kspace_mask.shape[1:],
                    mask=sim.kspace_mask[i],
                    n_coils=sim.n_coils,
                    smaps=sim.smaps,
                    **kwargs,
                )
                for i in range(len(sim.kspace_data))
            ]
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

    name = "adjoint"

    def __str__(self):
        return "Adjoint"

    def setup(self, sim: SimData) -> None:
        """Set up the reconstructor."""
        self.reconstructor = get_fourier_operator(sim)

    def reconstruct(self, sim: SimData) -> np.ndarray:
        """Reconstruct with Zero filled method."""
        if self.reconstructor is None:
            self.setup(sim)
        return self.reconstructor.adj_op(sim.kspace_data)


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

    name = "sequential"

    def __init__(
        self,
        max_iter_per_frame: int = 15,
        optimizer: str = "pogm",
        wavelet: str = "sym8",
        threshold: float | Literal["sure"] = "sure",
    ):
        super().__init__()
        self.max_iter_per_frame = max_iter_per_frame
        self.optimizer = optimizer
        self.wavelet = wavelet
        self.threshold = threshold

    def setup(self, sim: SimData) -> None:
        """Set up the reconstructor."""
        from fmri.operators.wavelet import WaveletTransform
        from fmri.operators.weighted import AutoWeightedSparseThreshold
        from modopt.opt.proximity import SparseThreshold

        if self.fourier_op is None:
            self.fourier_op = get_fourier_operator(sim, repeat=True)

        space_linear_op = WaveletTransform(
            self.wavelet, shape=sim.shape, level=3, mode="periodization"
        )
        space_linear_op.op(np.zeros_like(sim.data_ref[0]))

        if self.threshold == "sure":
            space_prox_op = AutoWeightedSparseThreshold(
                space_linear_op.coeffs_shape,
                threshold_estimation="hybrid-sure",
                threshold_scaler=0.6,
            )
        else:
            self.threshold = float(self.threshold)
            space_prox_op = SparseThreshold(self.threshold)
        from fmri.reconstructors.frame_based import SequentialReconstructor

        self.reconstructor = SequentialReconstructor(
            self.fourier_op, space_linear_op, space_prox_op, optimizer="pogm"
        )

    def reconstruct(
        self, sim: SimData, fourier_op: SpaceFourierBase | None = None
    ) -> np.ndarray:
        """Reconstruct with Sequential."""
        if fourier_op is not None:
            self.fourier_op = fourier_op
        if self.reconstructor is None:
            self.setup(sim)

        seq_rec = self.reconstructor.reconstruct(
            sim.kspace_data, max_iter_per_frame=self.max_iter_per_frame, warm_x=True
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

    name = "lr+f"

    def __init__(
        self,
        lambda_l: float = 0.1,
        lambda_s: float | Literal["sure"] = 1,
        algorithm: str = "otazo",
        max_iter: int = 20,
        time_linear_op: LinearParent = None,
        time_prox_op: ProximityParent = None,
        space_prox_op: ProximityParent = None,
        fourier_op: SpaceFourierBase = None,
    ):
        super().__init__()
        self.lambda_l = lambda_l
        self.lambda_s = lambda_s
        self.max_iter = max_iter
        self.algorithm = algorithm

        self.time_linear_op = time_linear_op
        self.time_prox_op = time_prox_op
        self.space_prox_op = space_prox_op
        self.fourier_op = fourier_op

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
            self.fourier_op = get_fourier_operator(sim, repeat=False)

        if self.time_linear_op is None:
            self.time_linear_op = TimeFourier(time_axis=0)

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

        if self.space_prox_op is None:
            self.space_prox_op = FlattenSVT(
                self.lambda_l, initial_rank=10, thresh_type="soft-rel"
            )

        self.reconstructor = LowRankPlusSparseReconstructor(
            self.fourier_op,
            space_prox_op=self.space_prox_op,
            time_prox_op=self.time_prox_op,
            cost="auto",
        )

    def reconstruct(
        self, sim: SimData, fourier_op: SpaceFourierBase | None = None
    ) -> np.ndarray:
        """Reconstruct using LowRank+Sparse Method."""
        if fourier_op is not None:
            self.fourier_op = fourier_op
        if self.reconstructor is None:
            self.setup(sim)
        M, L, S, costs = self.reconstructor.reconstruct(
            sim.kspace_data,
            grad_step=0.5,
            max_iter=self.max_iter,
            optimizer=self.algorithm,
        )
        return M


class LowRankPlusTVReconstructor(LowRankPlusSparseReconstructor):
    ...


class LowRankPlusWaveletReconstructor(LowRankPlusSparseReconstructor):
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

    name = "adjoint+denoised"

    def __init__(
        self, patch_shape: int, patch_overlap: int, recombination: str = "weighted"
    ):
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
