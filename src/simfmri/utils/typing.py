"""Typing utilities."""
from __future__ import annotations

import numpy as np

RngType = int | np.random.Generator | None
"""Type characterising a random number generator.

A random generator is either reprensented by its seed (int),
or a numpy.random.Generator.
"""


Shape2d = tuple[int, int]
"""Type for a 2D shape."""
Shape3d = tuple[int, int, int]
"""Type for a 3D shape."""

AnyShape = Shape2d | Shape3d

"""Type for a 2D or 3D shape."""
