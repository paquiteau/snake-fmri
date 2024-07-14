"""Operators for Cartesian MRI."""

from __future__ import annotations
from numpy.typing import NDArray

import numpy as np
import scipy as sp


def fft(image: NDArray, axis: tuple[int] = -1) -> NDArray:
    """Apply the FFT operator.

    Parameters
    ----------
    image : array
        Image in space.
    axis : int
        Axis to apply the FFT.

    Returns
    -------
    kspace_data : array
        kspace data.
    """
    return sp.fft.ifftshift(
        sp.fft.fftn(sp.fft.fftshift(image, axes=axis), norm="ortho", axes=axis),
        axes=axis,
    )


def ifft(kspace_data: NDArray, axis: tuple[int] = -1) -> NDArray:
    """Apply the inverse FFT operator."""
    return sp.fft.fftshift(
        sp.fft.ifftn(sp.fft.ifftshift(kspace_data, axes=axis), norm="ortho", axes=axis),
        axes=axis,
    )


class FFT_MRI:
    """Apply the FFT with potential Smaps support.

    Parameters
    ----------
    shape : tuple
        Shape of the image.
    n_coils : int
        Number of coils.
    n_batch: int
    mask : array
        Mask of the image.
    smaps : array
        Sensitivity maps.
    """

    def __init__(
        self,
        shape: tuple[int],
        n_coils: int,
        mask: NDArray,
        n_batchs: int = 1,
        smaps: NDArray = None,
    ):
        self.shape = shape
        self.n_coils = n_coils
        self.n_batchs = n_batchs
        self.mask = mask
        self.smaps = smaps

    @property
    def uses_sense(self) -> bool:
        """True if operator uses smaps."""
        return self.smaps is not None

    def op(self, img: NDArray) -> NDArray:
        """Apply the forward operator."""
        axes = tuple(range(-len(self.shape), 0))
        if self.n_coils > 1:
            if self.uses_sense:
                img = img.reshape(self.n_batchs, 1, *self.shape)
                img2 = np.repeat(img, self.n_coils, axis=1).astype(np.complex64)
                img2 *= self.smaps
                ksp = fft(img2, axis=axes)
            else:
                img = img.reshape(self.n_batchs, self.n_coils, *self.shape)
                ksp = fft(img, axis=axes)
            return ksp * self.mask[None, None, ...]
        else:
            return fft(img, axis=axes) * self.mask[None, None, ...]

    def adj_op(self, ksp: NDArray) -> NDArray:
        """Apply the adjoint operator."""
        axes = tuple(range(-len(self.shape), 0))
        if self.n_coils > 1:
            img = ifft(ksp, axis=axes)
            if self.uses_sense:
                return img
            return np.sum(img * np.conj(self.smaps), axis=0)
        else:
            return ifft(ksp, axis=axes)
