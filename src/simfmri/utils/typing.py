"""Typing utilities."""
from __future__ import annotations

import numpy as np

RngType = int | np.random.Generator | None
"""Type characterising a random number generator.

A random generator is either reprensented by its seed (int),
or a numpy.random.Generator.
"""

AnyShape = tuple[int, ...]
