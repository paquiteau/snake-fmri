"""
Simulation data model.

The Simulation class holds all the information and data relative to a simulation.
"""

from __future__ import annotations
from typing import Literal, Any, TypeVar
import copy
import pickle
import dataclasses

import numpy as np
from numpy.typing import DTypeLike, NDArray
from simfmri.utils import AnyShape, cplx_type
from .lazy import LazySimArray

__all__ = ["SimulationData", "SimulationParams", "LazySimArray", "SimDataType"]

DataArray = NDArray[np.float64 | np.complex128]


@dataclasses.dataclass
class SimulationParams:
    """Simulation metadata."""

    shape: AnyShape
    """Shape of the volume of the simulation."""
    n_frames: int
    """Number of frame of the simulation."""
    sim_tr: float
    """Time resolution for the simulation."""
    n_coils: int = 1
    """Number of coil of the simulation."""
    rng: int = 19980408
    """Random number generator seed."""
    extra_infos: dict[str, Any] = dataclasses.field(
        default_factory=lambda: dict(), repr=False
    )
    """Extra information, to add more information to the simulation"""
    res: tuple[float, ...] = 1
    """Resolution of the volume in mm"""

    def __post_init__(self) -> None:
        if isinstance(self.res, (float, int)):
            self.res = (self.res,) * len(self.shape)


class SimulationData:
    """Data container for a simulation.

    Parameters
    ----------
    shape
        Shape of the volume simulated
    n_frames
        Number of frames acquired
    sim_tr
        Acquisition time for one volume
    n_coils
        Number of coils acquired, default 1
    rng
        Random number generator seed
    **extra_infos
        dict

    Attributes
    ----------
    shape
        Shape of the volume simulated
    n_frames
        Number of frames acquired
    sim_tr
        Time resolution for the simulation.
    n_coils
        Number of coils acquired, default 1
    rng
        Random number generator seed
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

    static_vol: DataArray
    data_ref: DataArray
    roi: DataArray
    _data_acq: DataArray
    data_rec: DataArray
    kspace_data: DataArray
    kspace_mask: DataArray
    kspace_location: DataArray
    smaps: DataArray | None

    def __init__(
        self,
        shape: AnyShape,
        res: float | tuple[float],
        sim_tr: float,
        sim_time: float,
        n_coils: int = 1,
        rng: int = 19980408,
        extra_infos: dict[str, Any] | None = None,
        lazy: bool = False,
    ) -> None:
        n_frames = int(sim_time / sim_tr)
        if extra_infos is None:
            extra_infos = dict()
        self._meta = SimulationParams(
            shape,
            res=res,
            n_frames=n_frames,
            sim_tr=sim_tr,
            n_coils=n_coils,
            rng=rng,
            extra_infos=extra_infos,
        )
        self.static_vol = None
        self.data_ref = None
        self.roi = None
        self._data_acq = None
        self.data_rec = None
        self.kspace_data = None
        self._data_acq = None
        self.data_ref = None
        self.kspace_mask = None
        self.kspace_location = None
        self.smaps = None
        self.lazy = lazy

    @classmethod
    def from_params(
        cls, sim_meta: SimulationParams, in_place: bool = False
    ) -> SimDataType:
        """Create a Simulation from its meta parameters.

        Parameters
        ----------
        sim_meta
            The meta parameters structure, it must be convertible to a
            dict with ``shape, TR, n_frames, n_coils`` attributes.
        in_place
            If True, the underlying _meta attribute is set to sim_meta.
        """
        if isinstance(sim_meta, SimulationParams):
            obj = cls(**dataclasses.asdict(sim_meta))
        else:
            obj = cls(**dict(sim_meta))
        if in_place and isinstance(sim_meta, SimulationParams):
            obj._meta = sim_meta

        return obj

    @classmethod
    def load_from_file(cls, filename: str, dtype: DTypeLike) -> SimDataType:
        """Load a simulation from file.

        Parameters
        ----------
        filename
            location of the stored Simulation.
        dtype
            The dtype
        """
        with open(filename, "rb") as f:
            obj: SimDataType = pickle.load(f)
        if obj.is_valid():
            for attr in obj.__dict__:
                val = getattr(obj, attr)
                if attr != "roi" and isinstance(val, np.ndarray):
                    if val.dtype in [np.complex64, np.complex128]:
                        cdtype = cplx_type(dtype)
                    else:
                        cdtype = dtype
                    setattr(obj, attr, val.astype(cdtype))

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
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def copy(self) -> SimDataType:
        """Return a deep copy of the Simulation."""
        return copy.deepcopy(self)

    @property
    def duration(self) -> float:
        """Return the duration (in seconds) of the experiment."""
        return self.sim_tr * self.n_frames

    @property
    def data_acq(self) -> np.ndarray:
        """Return the defacto acquired data if defined, else the reference data."""
        if self._data_acq is not None:
            return self._data_acq
        return self.data_ref

    @data_acq.setter
    def data_acq(self, value: np.ndarray) -> None:
        """Set the acquired data."""
        self._data_acq = value

    @property
    def shape(self) -> AnyShape:
        """Get shape."""
        return self._meta.shape

    @property
    def n_frames(self) -> int:
        """Get number of frames."""
        return self._meta.n_frames

    @property
    def res(self) -> int:
        """Get resolution."""
        return self._meta.res

    @property
    def sim_tr(self) -> float:
        """Get TR in seconds."""
        return self._meta.sim_tr

    @property
    def sim_tr_ms(self) -> int:
        """Get TR in milliseconds."""
        return int(self._meta.sim_tr * 1000)

    @property
    def sim_time(self) -> float:
        """Get the total simulation time."""
        return self.n_frames * self.sim_tr

    @property
    def sim_time_ms(self) -> int:
        """Get the total simulation time in milliseconds."""
        return int(self.n_frames * self.sim_tr * 1000)

    def get_sample_time(self, unit: Literal["s", "ms"] = "s") -> Any:  # noqa: ANN401
        """Get the time vector of the simulation."""
        if unit == "s":
            return np.arange(0, self.n_frames) * self.sim_tr
        elif unit == "ms":
            return np.arange(0, self.n_frames) * self.sim_tr_ms
        else:
            raise ValueError("unit must be s or ms")

    @property
    def n_coils(self) -> int:
        """Get number of coils."""
        return self._meta.n_coils

    @property
    def extra_infos(self) -> dict[str, Any]:
        """Get extra infos."""
        return self._meta.extra_infos

    @property
    def rng(self) -> int:
        """Get the random number generator seed."""
        return self._meta.rng

    @property
    def meta(self) -> SimulationParams:
        """Get meta Parameters."""
        return self._meta

    @meta.setter
    def meta(self, value: SimulationParams) -> None:
        if not isinstance(value, SimulationParams):
            raise ValueError("meta must be a SimulationParams object")
        self._meta = value

    def is_valid(self) -> bool:
        """Check if the attributes are coherent to each other."""
        if self.data_ref is not None:
            if self.data_ref.shape != (self.n_frames, *self.shape):
                return False
            if self.data_acq.shape != self.data_ref.shape:
                return False

        if self.smaps:
            if self.smaps.shape != (self.n_coils, *self.shape):
                return False
        if self.kspace_data is not None:
            if self.kspace_data.shape[0] != self.n_frames:
                return False
        if self.kspace_data is not None and self.smaps is not None:
            if self.kspace_data.shape[1] != self.n_coils:
                return False
        # TODO: add test on the mask

        return True

    def memory_estimate(self, unit: str = "MB") -> int | float:
        """Return an estimate of the memory used by the simulation."""
        mem = 0
        for attr in self.__dict__:
            val = getattr(self, attr)
            if isinstance(val, np.ndarray):
                mem += val.nbytes
        if self.lazy:
            arr = self.data_acq
            while isinstance(arr, LazySimArray):
                arr = arr._base_array
            mem += arr.nbytes * self.n_frames

        if unit == "MB":
            return mem / 1024**2
        elif unit == "GB":
            return mem / 1024**3
        else:
            raise ValueError("unit must be MB or GB")

    def __str__(self) -> str:
        ret = "SimulationData: \n"
        ret += f"{self._meta}\n"

        for array_name in [
            "data_ref",
            "data_acq",
            "kspace_data",
            "kspace_mask",
            "roi",
            "smaps",
        ]:
            array = getattr(self, array_name)
            if isinstance(array, np.ndarray):
                ret += f"{array_name}: {array.dtype}({array.shape})\n"
            else:
                ret += f"{array_name}: {array}\n"

        ret += "extra_infos:\n"
        for k, v in self.extra_infos.items():
            if isinstance(v, np.ndarray):
                ret += f" {k}: {v.dtype}({v.shape})\n"
            else:
                ret += f" {k}: {v}\n"
        return ret


SimDataType = TypeVar("SimDataType", bound=SimulationData)
