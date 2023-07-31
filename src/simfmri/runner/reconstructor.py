"""Reconstructor interface.

Reconstructor implement a `reconstruct` method which takes a simulation object as input
and returns a reconstructed fMRI array.
"""
from __future__ import annotations
from typing import Literal
import logging

import numpy as np
from mrinufft.operators import get_operator
from simfmri.simulator import SimulationData

from fmri.operators.fourier import FFT_Sense, RepeatOperator
from fmri.operators.fourier import CartesianSpaceFourier

logger = logging.getLogger(__name__)


def get_fourier_operator(
    sim: SimulationData, repeat: bool = False, **kwargs: None
) -> RepeatOperator | CartesianSpaceFourier:
    """Return a Fourier operator for the given simulation."""
    kwargs = kwargs.copy() if kwargs is not None else {}

    if backend := sim.extra_infos.get("operator", None):
        if "finufft" in backend:
            kwargs["squeeze_dims"] = True
        kwargs["density"] = True

        def _get_op(i: int = 0) -> RepeatOperator | CartesianSpaceFourier:
            return get_operator(backend)(
                sim.kspace_mask[i],
                shape=sim.shape,
                n_coils=sim.n_coils,
                smaps=sim.smaps,
                **kwargs,
            )

        if sim.extra_infos["traj_constant"]:
            return RepeatOperator([_get_op(0)] * len(sim.kspace_data))
        else:
            return RepeatOperator([_get_op(i) for i in range(len(sim.kspace_data))])
    elif repeat:
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


class BenchmarkReconstructor:
    """Represents the interface required to be benchmark-able."""

    def __init__(self):
        self.reconstructor = None

    def setup(self, sim: SimulationData) -> None:
        """Set up the reconstructor."""
        logger.info(f"Setup reconstructor {self.__class__.__name__}")

    def reconstruct(self, sim: SimulationData) -> np.ndarray:
        """Reconstruct data."""
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class ZeroFilledReconstructor(BenchmarkReconstructor):
    """Reconstruction using zero-filled (ifft) method."""

    def __str__(self):
        return "ZeroFilled"

    def setup(self, sim: SimulationData) -> None:
        """Set up the reconstructor."""
        self.reconstructor = get_fourier_operator(sim)

    def reconstruct(self, sim: SimulationData) -> np.ndarray:
        """Reconstruct with Zero filled method."""
        if self.reconstructor is None:
            self.setup(sim)
        return self.reconstructor.adj_op(sim.kspace_data)


class SequentialReconstructor(BenchmarkReconstructor):
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

    def setup(self, sim: SimulationData) -> None:
        """Set up the reconstructor."""
        from mri.operators.linear.wavelet import WaveletN
        from mri.operators.proximity.weighted import AutoWeightedSparseThreshold
        from modopt.opt.proximity import SparseThreshold

        fourier_op = get_fourier_operator(sim, repeat=True)

        space_linear_op = WaveletN(
            self.wavelet, nb_scale=3, dim=len(sim.shape), padding="periodization"
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
            fourier_op, space_linear_op, space_prox_op, optimizer="pogm"
        )

    def reconstruct(self, sim: SimulationData) -> np.ndarray:
        """Reconstruct with Sequential."""
        if self.reconstructor is None:
            self.setup(sim)
        seq_rec = self.reconstructor.reconstruct(
            sim.kspace_data, max_iter_per_frame=self.max_iter_per_frame, warm_x=True
        )
        return seq_rec


class LowRankPlusSparseReconstructor(BenchmarkReconstructor):
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

    def __init__(
        self,
        lambda_l: float = 0.1,
        lambda_s: float | Literal["sure"] = 1,
        algorithm: str = "otazo",
        max_iter: int = 20,
    ):
        super().__init__()
        self.lambda_l = lambda_l
        self.lambda_s = lambda_s
        self.max_iter = max_iter
        self.algorithm = algorithm

    def __str__(self):
        return f"LRS-{self.lr_thresh:.2e}-{self.sparse_thresh:.2e}"

    def setup(self, sim: SimulationData) -> None:
        """Set up the reconstructor."""
        from fmri.reconstructors.time_aware import LowRankPlusSparseReconstructor
        from fmri.operators.time_op import TimeFourier
        from fmri.operators.utils import sure_est, sigma_mad
        from fmri.operators.svt import FlattenSVT
        from fmri.operators.utils import InTransformSparseThreshold

        fourier_op = get_fourier_operator(sim, repeat=False)
        time_linear_op = TimeFourier(time_axis=0)
        space_prox_op = FlattenSVT(
            self.lambda_l, initial_rank=10, thresh_type="soft-rel"
        )

        if self.lambda_s == "sure":
            adj_data = fourier_op.adj_op(sim.kspace_data)
            sure_thresh = np.zeros(np.prod(adj_data.shape[1:]))
            tf = time_linear_op.op(adj_data).reshape(len(adj_data), -1)
            for i in range(len(sure_thresh)):
                sure_thresh[i] = sure_est(tf[:, i]) * sigma_mad(tf[:, i])

            self.lambda_s = np.median(sure_thresh) / 2
            logger.info(f"SURE threshold: {self.lambda_s:.4e}")

        time_prox_op = InTransformSparseThreshold(
            time_linear_op, self.lambda_s, thresh_type="soft"
        )

        self.reconstructor = LowRankPlusSparseReconstructor(
            fourier_op,
            space_prox_op=space_prox_op,
            time_prox_op=time_prox_op,
            cost="auto",
        )

    def reconstruct(self, sim: SimulationData) -> np.ndarray:
        """Reconstruct using LowRank+Sparse Method."""
        if self.reconstructor is None:
            self.setup(sim)
        M, L, S, costs = self.reconstructor.reconstruct(
            sim.kspace_data,
            grad_step=0.5,
            max_iter=self.max_iter,
            optimizer=self.algorithm,
        )
        return M


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

    def reconstruct(self, sim: SimulationData) -> np.ndarray:
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


# TODO: PnP
