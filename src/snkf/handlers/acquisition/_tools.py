import numpy as np

from typing import Generator, Protocol


SimGeneratorType = Generator[np.ndarray, None, None]
TrajectoryGeneratorType = Generator[np.ndarray, None, None]


class TrajectoryFactoryProtocol(Protocol):
    """Protocol for trajectory factory."""

    def __call__(self, shape: tuple[int, ...], **__kwargs: None) -> np.ndarray:
        """Create a trajectory."""
        ...
