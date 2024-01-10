import numpy as np

from typing import Generator, Protocol, Callable

from simfmri.utils.typing import AnyShape

SimGeneratorType = Generator[np.ndarray, None, None]
TrajectoryGeneratorType = Generator[np.ndarray, None, None]


class TrajectoryFactoryProtocol(Protocol):
    """Protocol for trajectory factory."""

    def __call__(self, shape: AnyShape, **__kwargs: None) -> np.ndarray:
        """Create a trajectory."""
        ...
