#!/usr/bin/env python3

import numpy as np


def validate_rng(rng=None):
    """Validate Random Number Generator."""
    if isinstance(rng, int):
        return np.random.default_rng(rng)
    elif rng is None:
        return np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        return rng
    else:
        raise ValueError("rng shoud be a numpy Generator, None or an integer seed.")
