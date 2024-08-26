"""FFT operators for MRI reconstruction."""

import scipy as sp


def fft(image, axis=-1):
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


def ifft(kspace_data, axis=-1):
    """Apply the inverse FFT operator."""
    return sp.fft.fftshift(
        sp.fft.ifftn(sp.fft.ifftshift(kspace_data, axes=axis), norm="ortho", axes=axis),
        axes=axis,
    )
