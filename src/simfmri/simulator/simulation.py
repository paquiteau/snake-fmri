"""
Simulation data model.

The Simulation class holds all the information and data relative to a simulation.
"""
from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass


from simfmri.utils import Shape2d3dType


@dataclass
class SimulationMeta:
    """Simulation metadata."""

    shape: Shape2d3dType
    """Shape of the volume of the simulation."""
    n_frames: int
    """Number of frame of the simulation."""
    TR: float
    """Samping time."""
    n_coils: int = 1
    """Number of coil of the simulation."""
    extras_infos: dict = None
    """Extra information, to add more information to the simulation"""


class Simulation:
    """Data container for a simulation.

    Parameters
    ----------
    shape
        Shape of the volume simulated
    n_frames
        Number of frames acquired
    TR
        Acquisition time for one volume/frame
    n_coils
        Number of coils acquired, default 1
    **extra_infos
        dict

    Attributes
    ----------
    shape
        Shape of the volume simulated
    n_frames
        Number of frames acquired
    TR
        Acquisition time for one volume/frame
    n_coils
        Number of coils acquired, default 1
    extra_infos: dict
        Extra information store in a dictionnary.
    static_vol: np.ndarray
        Static representation of the volume, eg anatomical T2
    data_ref: np.ndarray = None
        Simulation data array with shape (n_frames, *shape).
        This data should remain noise free !
    roi: np.ndarray = None
        Array of shape shape defining the region where activation occurs.
        It can be either a boolean array, or a float array with values between 0 and 1.
    data_acq: np.ndarray = None
        Image data that would be acquired by the scanner.
    data_rec: np.ndarray = None
        Image data after reconstruction
    kspace_data: np.ndarray = None
        Kspace data available for reconstruction.
        It has the shape (n_frames, n_coils, kspace_dims)
    kspace_mask: np.ndarray = None
        Mask of the sample kspace data shape (n_frames, kspace_dims)
    kspace_location: np.ndarray = None
        Location of kspace samples., shape (n_frames, kspace_dims)
    smaps: np.ndarray = None
        If n_coils > 1 , describes the sensitivity maps of each coil.
    """

    def __init__(
        self,
        shape: Shape2d3dType,
        n_frames: int,
        TR: float,
        n_coils: int = 1,
        **extra_infos,
    ) -> Simulation:
        self._meta = SimulationMeta(
            shape, n_frames, TR, n_coils, extra_infos=extra_infos
        )

        self.static_vol = None
        self.data_ref = None
        self.roi = None
        self.data_acq = None
        self.datat_rec = None
        self.kspace_data = None
        self.kspace_mask = None
        self.kspace_location = None
        self.smaps = None

    @classmethod
    def from_meta(cls, sim_meta: SimulationMeta, in_place=False) -> Simulation:
        """Create a Simulation from its meta parameters.

        Parameters
        ----------
        sim_meta
            The meta parameters structure, it must be convertible to a
            dict with ``shape, TR, n_frames, n_coils`` attributes.
        in_place
            If True, the underlying _meta attribute is set to sim_meta.
        """
        obj = cls(**dict(sim_meta))
        if in_place and isinstance(sim_meta, SimulationMeta):
            obj._meta = sim_meta

        return obj

    @classmethod
    def load_from_file(cls, filename: str) -> Simulation:
        """Load a simulation from file.

        Parameters
        ----------
        filename
            location of the stored Simulation.
        """
        with open(filename) as f:
            obj = pickle.load(f)
        if obj.is_valid():
            return obj
        else:
            raise ValueError("Simulation object not valid.")

    def save(self, filename: str) -> None:
        """
        Save the simulation to file.

        Parameters
        ----------
        filename
        """
        with open(filename) as f:
            pickle.dump(self, f)

    def copy(self) -> Simulation:
        """Return a deep copy of the Simulation."""
        return copy.deepcopy(self)

    @property
    def duration(self) -> float:
        """Return the duration (in seconds) of the experiment."""
        return self.TR * self.n_frames

    def is_valid(self) -> bool:
        """Check if the attributes are coherent to each other."""
        if self.data_ref:
            if self.data_ref.shape != (self.n_frames, *self.shape):
                return False
            if self.data_acq.shape != self.data_ref.shape:
                return False

        if self.smaps:
            if self.smaps.shape != (self.n_coils, *self.shape):
                return False
        if self.kspace_data:
            if self.kspace_data.shape[0] != self.n_frames:
                return False
        if self.kspace_data and self.smaps:
            if self.kspace_data.shape[1] != self.n_coils:
                return False
        # TODO: add test on the mask

        return True


# expose the meta attribute at first level as read-only.
for attr in SimulationMeta.__dict__:
    setattr(Simulation, attr, property(lambda obj, attr=attr: getattr(obj._meta, attr)))
