from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import copy


@dataclass
class Simulation:
    """Data container for a simulation."""

    shape: tuple
    """Shape of the volume of the simulation """
    n_frames: int
    """Number of frame of the simulation."""
    TR: float
    """Samping time"""
    n_coils: int = 1
    """Number of coil of the simulation"""
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
    """Kspace data available for reconstruction.
    shape= (n_frames, n_coils, kspace_dims)"""
    kspace_mask: np.ndarray = None
    """Mask of the sample kspace data shape (n_frames, kspace_dims)"""
    kspace_location: np.ndarray = None
    """Location of kspace samples., shape (n_frames, kspace_dims)"""
    smaps: np.ndarray = None
    """If n_coils > 1 , describes the sensitivity maps of each coil."""

    extras_infos: dict = None
    """Extra information, to add more information to the simulation"""

    def load_from_file(filename):
        """Load a simulation from file."""
        raise NotImplementedError

    def save(filename):
        """Save a simulation to file."""

    def copy(self):
        """Return a deep copy of the Simulation."""
        return copy.deepcopy(self)

    @property
    def duration(self):
        return self.TR * self.n_frames
