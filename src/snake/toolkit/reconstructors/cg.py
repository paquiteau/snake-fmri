"""Conjugate Gradient descent solver."""

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from snake.mrd_utils import (
    CartesianFrameDataLoader,
    NonCartesianFrameDataLoader,
)

from .fourier import init_nufft
from .pysap import ZeroFilledReconstructor


class ConjugateGradientReconstructor(ZeroFilledReconstructor):
    """Conjugate Gradient descent solver.

    Parameters
    ----------
    max_iter : int
            Maximum number of iterations.
    tol : float
            Tolerance for the solver.
    """

    __reconstructor_name__ = "cg"

    max_iter: int = 50
    tol: float = 1e-4
    density_compensation: str | bool | None = False
    nufft_backend: str = "cufinufft"

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return f"{self.__reconstructor_name__}_{self.max_iter}_{self.tol:.0e}"

    def _reconstruct_nufft(self, data_loader: NonCartesianFrameDataLoader) -> NDArray:
        """Reconstruct the data using the NUFFT operator."""
        from mrinufft.extras.gradient import cg

        nufft_operator = init_nufft(
            data_loader,
            density_compensation=self.density_compensation,
            nufft_backend=self.nufft_backend,
        )
        final_images = np.empty(
            (data_loader.n_frames, *data_loader.shape), dtype=np.complex64
        )

        for i in tqdm(range(data_loader.n_frames)):
            traj, data = data_loader.get_kspace_frame(i)
            if data_loader.slice_2d:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                )[0, :, :2]
                data = np.reshape(data, (data.shape[0], data_loader.n_shots, -1))
                for j in range(data.shape[1]):
                    final_images[i, :, :, j] = cg(
                        nufft_operator, data[:, j],num_iter=self.max_iter, tol=self.tol
                        )
            else:
                final_images[i] = cg(
                    nufft_operator, data, num_iter=self.max_iter, tol=self.tol
                )
        return final_images

    def _reconstruct_cartesian(self, data_loader: CartesianFrameDataLoader) -> NDArray:
        """Reconstruct the data for Cartesian Settings."""
        from mrinufft.extras.fft import (
            CartesianFourierOperator,
        )  # TODO this does not exists yet
        from mrinufft.extras.gradient import cg

        mask, data = data_loader.get_kspace_frame(0)
        nufft_operator = CartesianFourierOperator(mask, data_loader.shape)

        final_images = np.empty(
            (data_loader.n_frames, *data_loader.shape), dtype=np.float32
        )

        for i in tqdm(range(data_loader.n_frames)):
            traj, data = data_loader.get_kspace_frame(i)
            if data_loader.slice_2d:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                )[0, :, :2]
                data = np.reshape(data, (data.shape[0], data_loader.n_shots, -1))
                for j in range(data.shape[1]):
                    final_images[i, :, :, j] = cg(nufft_operator, data[:, j])
            else:
                final_images[i] = cg(
                    nufft_operator, data, num_iter=self.max_iter, tol=self.tol
                )
        return final_images
