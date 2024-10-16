"""FFT operators for MRI reconstruction."""

from numpy.typing import NDArray
import scipy as sp


def fft(image: NDArray, axis: int | tuple[int] = -1) -> NDArray:
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


def ifft(kspace_data: NDArray, axis: int | tuple[int] = -1) -> NDArray:
    """Apply the inverse FFT operator.

    Parameters
    ----------
    kspace_data : array
        Image in space.
    axis : int
        Axis to apply the FFT.

    Returns
    -------
    image_data : array
        image data.
    """
    return sp.fft.fftshift(
        sp.fft.ifftn(sp.fft.ifftshift(kspace_data, axes=axis), norm="ortho", axes=axis),
        axes=axis,
    )
