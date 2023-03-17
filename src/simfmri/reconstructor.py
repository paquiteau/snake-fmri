"""Reconstructor interface.

Reconstructor implement a `reconstruct` method which takes a simulation object as input
and returns a reconstructed fMRI array.
"""
import numpy as np

from simfmri.simulator import SimulationData

from fmri.operators.fourier import SpaceFourierBase, CartesianSpaceFourier


class BenchmarkReconstructor:
    """Represents the interface required to be benchmark-able."""

    def reconstruct(self, sim: SimulationData) -> np.ndarray:
        """Reconstruct data."""
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


def get_fourier_operator(sim: SimulationData) -> SpaceFourierBase:  # noqa ANN201
    """Get fourier operator from from the simulation.

    Parameters
    ----------
    sim
        Simulation Data that contains all the information to create a fourier operator.
    TODO: add support for non cartesian simulation.
    """
    return CartesianSpaceFourier(
        shape=sim.shape,
        mask=sim.kspace_mask,
        n_frames=sim.n_frames,
        n_coils=sim.n_coils,
        smaps=sim.smaps,
    )


class ZeroFilledReconstructor(BenchmarkReconstructor):
    """Reconstruction using zero-filled (ifft) method."""

    def __str__(self):
        return "ZeroFilled"

    def reconstruct(self, sim: SimulationData) -> np.ndarray:
        """Reconstruct with Zero filled method."""
        fourier_op = get_fourier_operator(sim)
        return fourier_op.adj_op(sim.kspace_data)


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
        threshold: float = 2e-7,
    ):
        self.max_iter_per_frame = max_iter_per_frame
        self.optimizer = optimizer
        self.threshold = threshold
        self.wavelet = wavelet

    def __str__(self):
        return (
            f"Sequential-{self.max_iter_per_frame}iter"
            f"-{self.optimizer}-{self.threshold}"
        )

    def reconstruct(self, sim: SimulationData) -> np.ndarray:
        """Reconstruct with Sequential."""
        from fmri.reconstructors.frame_based import SequentialFMRIReconstructor
        from modopt.opt.linear import Identity
        from modopt.opt.proximity import SparseThreshold
        from mri.operators.linear.wavelet import WaveletN

        fourier_op = get_fourier_operator(sim)
        wavelet_op = WaveletN(self.wavelet, dim=len(sim.shape))
        # warmup
        wavelet_op.op(np.zeros(sim.shape))

        sec_rec = SequentialFMRIReconstructor(
            fourier_op,
            space_linear_op=wavelet_op,
            space_prox_op=SparseThreshold(
                Identity(),
                self.threshold,
                thresh_type="hard",
            ),
            optimizer=self.optimizer,
            progbar_disable=True,
        )
        return sec_rec.reconstruct(
            sim.kspace_data, max_iter_per_frame=self.max_iter_per_frame
        )


class LowRankPlusSParseReconstructor(BenchmarkReconstructor):
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
        lr_thresh: float = 0.1,
        sparse_thresh: float = 1,
        max_iter: int = 150,
        max_iter_frame: int = None,
    ):
        self.lr_thresh = lr_thresh
        self.sparse_thresh = sparse_thresh
        self.max_iter = max_iter
        self._max_iter_frame = max_iter_frame

    def __str__(self):
        return f"LRS-{self.lr_thresh:.2e}-{self.sparse_thresh:.2e}"

    def reconstruct(self, sim: SimulationData) -> np.ndarray:
        """Reconstruct using LowRank+Sparse Method."""
        from fmri.reconstructors.time_aware import LowRankPlusSparseReconstructor
        from modopt.opt.linear import Identity
        from modopt.opt.proximity import SparseThreshold
        from fmri.operators.svt import FlattenSVT

        if self._max_iter_frame is not None:
            max_iter = self._max_iter_frame * sim.n_frames
        else:
            max_iter = self.max_iter
        fourier_op = get_fourier_operator(sim)
        lowrank_op = FlattenSVT(
            threshold=self.lr_thresh, initial_rank=5, thresh_type="hard-rel"
        )
        # lowrank_op = FlattenRankConstraint(rank=1)
        sparse_op = SparseThreshold(
            linear=Identity(), weights=self.sparse_thresh, thresh_type="hard"
        )

        glrs = LowRankPlusSparseReconstructor(
            fourier_op=fourier_op, lowrank_op=lowrank_op, sparse_op=sparse_op
        )
        glrs_final = glrs.reconstruct(sim.kspace_data, max_iter=max_iter)[0]

        return glrs_final


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
