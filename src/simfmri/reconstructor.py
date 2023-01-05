"""Reconstructor interface.

Reconstructor implement a `reconstruct` method which takes a simulation object as input
and returns a reconstructed fMRI array.
"""
import numpy as np

from simfmri.simulator import SimulationData


class BenchmarkReconstructor:
    """Represents the interface required to be benchmark-able."""

    def reconstruct(self, sim: SimulationData) -> np.ndarray:
        """Reconstruct data."""
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class ZeroFilledReconstructor(BenchmarkReconstructor):
    """Reconstruction using zero-filled (ifft) method."""

    def __str__(self):
        return "ZeroFilled"

    def reconstruct(self, sim: SimulationData) -> np.ndarray:
        """Reconstruct with Zero filled."""
        from fmri.operators.fourier import CartesianSpaceFourier

        fourier_op = CartesianSpaceFourier(
            shape=sim.shape,
            mask=sim.kspace_mask,
            n_frames=sim.n_frames,
            n_coils=sim.n_coils,
            smaps=sim.smaps,
        )
        return fourier_op.adj_op(sim.kspace_data)


class SequentialReconstructor(BenchmarkReconstructor):
    """Use a sequential Reconstruction.

    Parameters
    ----------
    max_iter_frame
        Number of iteration to allow per frame.

    """

    def __init__(self, max_iter_per_frame: int = 15, optimizer="pogm", threshold=2e-7):
        self.max_iter_per_frame = max_iter_per_frame
        self.optimizer = optimizer
        self.threshold = threshold

    def __str__(self):
        return (
            f"Sequential-{self.max_iter_per_frame}iter"
            f"-{self.optimizer}-{self.threshold}"
        )

    def reconstruct(self, sim: SimulationData):
        """Reconstruct with Sequential."""
        from fmri.operator.fourier import CartesianSpaceFourier
        from fmri.reconstructors.frame_based import SequentialFMRIReconstructor
        from modopt.opt.linear import Identity
        from modopt.opt.proximity import SparseThreshold
        from mri.operators.linear.wavelet import WaveletN

        fourier_op = CartesianSpaceFourier(
            shape=sim.shape,
            mask=sim.kspace_mask,
            n_frames=sim.n_frames,
            n_coils=sim.n_coils,
            smaps=sim.smaps,
        )
        wavelet_op = WaveletN("sym8", dim=len(sim.shape))
        # warmup
        wavelet_op.op(np.zeros(sim.shape))

        sec_rec = SequentialFMRIReconstructor(
            fourier_op,
            space_linear_op=wavelet_op,
            space_prox_op=SparseThreshold(Identity(), 2 * 1e-7, thresh_type="soft"),
            optimizer=self.optimizer,
        )
        return sec_rec.reconstruct(
            sim.kspace_data, max_iter_per_frame=self.max_iter_per_frame
        )


# TODO: LR+S reconstructor
# TODO: ZeroFilled + Denosing
# TODO: PnP
