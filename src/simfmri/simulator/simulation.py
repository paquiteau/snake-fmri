from dataclasses import dataclass
from typing import Union, Tuple, TypeVar
import numpy as np


ShapeType = TypeVar(Union(Tuple(int, int, int), Tuple(int, int)))


@dataclass
class Simulation:
    """Data container for a simulation."""

    shape: ShapeType = None
    """Shape of the volume of the simulation """
    n_frames: int = 1
    """Number of frame of the simulation."""
    n_coils: int = 1
    """Number of coil of the simulation"""
    TR: float = 1.0
    """Temporal resolution"""
    static_vol: np.ndarray = None
    """Static representation of the volume, eg anatomical T2."""

    data_ref: np.ndarray = None
    """
    Simulation data array with shape (n_frames, *shape).
    This data should remain noise free !
    """
    roi: np.ndarray = None
    """
    Array of shape shape defining the region where activation occurs.
    It can be either a boolean array, or a float array where value are between 0 and 1.
    """
    data_acq: np.ndarray = None
    """Image data that would be acquired by the scanner."""
    kspace_data: np.ndarray = None
    """Kspace data available for reconstruction. shape= (n_frames, n_coils, *shape)"""
    kspace_mask: np.ndarray = None
    """Mask of the sample kspace data"""
    smaps: np.ndarray = None
    """If n_coils > 1 , describes the sensitivity maps of each coil."""

    def load_from_file(filename):
        """Load a simulation from file."""
        raise NotImplementedError

    def save(filename):
        """Save a simulation to file."""
