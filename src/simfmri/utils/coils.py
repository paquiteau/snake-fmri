"""Simulation of Smaps."""
import numpy as np


def get_smaps(shape, n_coils, antenna="birdcage", dtype=np.complex64):

    if antenna == "birdcage":
        return _birdcage_maps(shape, nzz=n_coils, dtype=dtype)
    else:
        raise NotImplementedError


def _birdcage_maps(shape, r=1.5, nzz=8, dtype=np.complex64):
    """Simulate birdcage coil sensitivies.

    Parameters
    ----------
    shape: tuple of int
        sensitivity maps shape (nc, x,y,z)
    r: float
        Relative radius of birdcage.
    nzz: int
        number of coils per ring.
    dtype: data type.

    Returns
    -------
    np.ndarray: complex sensitivity profiles.

    References
    ----------
    https://sigpy.readthedocs.io/en/latest/_modules/sigpy/mri/sim.html
    """
    nc, nz, ny, nx = shape
    c, z, y, x = np.mgrid[:nc, :nz, :ny, :nx]

    coilx = r * np.cos(c * (2 * np.pi / nzz), dtype=np.float32)
    coily = r * np.sin(c * (2 * np.pi / nzz), dtype=np.float32)
    coilz = np.floor(np.float32(c / nzz)) - 0.5 * (np.ceil(nc / nzz) - 1)
    coil_phs = np.float32(-(c + np.floor(c / nzz)) * (2 * np.pi / nzz))

    x_co = (x - nx / 2.0) / (nx / 2.0) - coilx
    y_co = (y - ny / 2.0) / (ny / 2.0) - coily
    z_co = (z - nz / 2.0) / (nz / 2.0) - coilz
    rr = (x_co**2 + y_co**2 + z_co**2) ** 0.5
    phi = np.arctan2(x_co, -y_co) + coil_phs
    out = (1 / rr) * np.exp(1j * phi)

    rss = sum(abs(out) ** 2, 0) ** 0.5
    out /= rss

    return out.astype(dtype)
