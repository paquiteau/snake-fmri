# /usr/bin/env python3
"""General utility tools."""
import logging
import numpy as np
from numpy.typing import DTypeLike

sim_logger = logging.getLogger("simulation")


def validate_rng(rng: int | np.random.Generator = None) -> np.random.Generator:
    """Validate Random Number Generator."""
    if isinstance(rng, int):
        return np.random.default_rng(rng)
    elif rng is None:
        return np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        return rng
    else:
        raise ValueError("rng shoud be a numpy Generator, None or an integer seed.")


def cplx_type(dtype: DTypeLike) -> DTypeLike:
    """Return the complex dtype with the same precision as a real one.

    Example
    -------
    >>> cplx_type(np.float32)
    np.complex64
    """
    d = np.dtype(dtype)
    if d.type is np.float64:
        return np.complex128
    elif d.type is np.float128:
        return np.complex256
    elif d.type is np.float32:
        return np.complex64
    else:
        sim_logger.warning(
            "not supported dtype, use matching complex64", stack_info=True
        )
        return np.complex64
