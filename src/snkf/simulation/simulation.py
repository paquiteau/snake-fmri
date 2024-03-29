"""
Simulation data model.

The Simulation class holds all the information and data relative to a simulation.
"""

from __future__ import annotations
from typing import Literal, Any, Union
import copy
import pickle
import logging

import numpy as np
from numpy.typing import DTypeLike, NDArray
from snkf.base import cplx_type
from snkf.config import SimParams
from .lazy import LazySimArray

logger = logging.getLogger(__name__)

__all__ = ["SimData", "SimParams"]

OptionalArray = Union[NDArray, None]


class UndefinedArrayError(ValueError):
    """Raise when an array is undefined."""


class SimData:
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

    def __init__(self, meta_params: SimParams) -> None:
        self._meta = meta_params
        self._static_vol: NDArray | None = None
        self._roi: NDArray | None = None
        self._data_ref: NDArray | None | LazySimArray = None
        self._data_acq: NDArray | None | LazySimArray = None
        self.data_rec: NDArray | None = None
        self._kspace_data: NDArray | None = None
        self._kspace_mask: NDArray | None = None
        self.smaps: NDArray | None = None

    @classmethod
    def from_params(
        cls,
        shape: tuple[int, int, int],
        fov: tuple[float],
        sim_tr: float,
        sim_time: float,
        n_coils: int = 1,
        rng: int = 19980408,
        extra_infos: dict[str, Any] | None = None,
        lazy: bool = False,
    ) -> SimData:
        """Initialize the Simulation directly with Parameters."""
        if extra_infos is None:
            extra_infos = dict()
        meta = SimParams(
            shape,
            fov=fov,
            sim_time=sim_time,
            sim_tr=sim_tr,
            n_coils=n_coils,
            rng=rng,
            extra_infos=extra_infos,
        )

        return cls(meta)

    @property
    def static_vol(self) -> NDArray:
        """Static volume."""
        if self._static_vol is not None:
            return self._static_vol
        raise UndefinedArrayError("static_vol is not defined")

    @static_vol.setter
    def static_vol(self, value: NDArray) -> None:
        self._static_vol = value

    @property
    def kspace_data(self) -> NDArray:
        """Static volume."""
        if self._kspace_data is not None:
            return self._kspace_data
        raise UndefinedArrayError("static_vol is not defined")

    @kspace_data.setter
    def kspace_data(self, value: NDArray) -> None:
        self._kspace_data = value

    @property
    def kspace_mask(self) -> NDArray:
        """Static volume."""
        if self._kspace_mask is not None:
            return self._kspace_mask
        raise UndefinedArrayError("static_vol is not defined")

    @kspace_mask.setter
    def kspace_mask(self, value: NDArray) -> None:
        self._kspace_mask = value

    @property
    def data_ref(self) -> NDArray | LazySimArray:
        """Static volume."""
        if self._data_ref is not None:
            return self._data_ref
        raise UndefinedArrayError("data_ref is not defined")

    @data_ref.setter
    def data_ref(self, value: NDArray) -> None:
        self._data_ref = value

    @property
    def data_acq(self) -> NDArray | LazySimArray:
        """Acquired Volume."""
        if self._data_acq is not None:
            return self._data_acq
        raise UndefinedArrayError("data_acq is not defined")

    @data_acq.setter
    def data_acq(self, value: NDArray) -> None:
        self._data_acq = value

    @property
    def roi(self) -> NDArray:
        """Reference data volume."""
        if self._roi is not None:
            return self._roi
        raise UndefinedArrayError("roi is not defined")

    @roi.setter
    def roi(self, value: NDArray) -> None:
        self._roi = value

    @classmethod
    def load_from_file(cls, filename: str, dtype: DTypeLike) -> SimData:
        """Load a simulation from file.

        Parameters
        ----------
        filename
            location of the stored Simulation.
        dtype
            The dtype
        """
        with open(filename, "rb") as f:
            obj: SimData = pickle.load(f)
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

    def copy(self) -> SimData:
        """Return a deep copy of the Simulation."""
        return copy.deepcopy(self)

    @property
    def duration(self) -> float:
        """Return the duration (in seconds) of the experiment."""
        return self.sim_tr * self.n_frames

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape."""
        return self._meta.shape

    @property
    def n_frames(self) -> int:
        """Get number of frames."""
        return self._meta.n_frames

    @property
    def hardware(self) -> Any:
        """Get number of frames."""
        return self._meta.hardware

    @property
    def fov(self) -> tuple[float, ...]:
        """Get the simulation FOV."""
        return self._meta.fov

    @property
    def res(self) -> tuple[float, ...]:
        """Get resolution."""
        return tuple(f / s for f, s in zip(self.fov, self.shape))

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
    def lazy(self) -> int:
        """Get Lazy status."""
        return self._meta.lazy

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
    def params(self) -> SimParams:
        """Get meta Parameters."""
        return self._meta

    @params.setter
    def params(self, value: SimParams) -> None:
        if not isinstance(value, SimParams):
            raise ValueError("params must be a SimulationParams object")
        self._meta = value

    def is_valid(self) -> bool:
        """Check if the attributes are coherent to each other."""
        try:
            data_ref = self.data_ref
        except UndefinedArrayError:
            logger.warn("data_ref is not defined yet.")
            data_ref = None
        if data_ref is not None:
            if self.data_ref.shape != (self.n_frames, *self.shape):
                logger.warn("self.data_ref.shape != (self.n_frames, *self.shape)")
                return False
            try:
                data_acq = self.data_acq
            except UndefinedArrayError:
                data_acq = None
            if data_acq is not None:
                if self.data_acq.shape != self.data_ref.shape:
                    logger.warn("self.data_acq.shape != self.data_ref.shape")
                return False

        if self.smaps is not None:
            if self.smaps.shape != (self.n_coils, *self.shape):
                logger.warn("self.smaps.shape != (self.n_coils, *self.shape)")
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
            if arr is not None:
                mem += arr.nbytes * self.n_frames
            else:
                raise ValueError("Unknown base array")
        if unit == "MB":
            return mem / 1024**2
        elif unit == "GB":
            return mem / 1024**3
        else:
            raise ValueError("unit must be MB or GB")

    def __str__(self) -> str:
        ret = "SimData: \n"
        ret += f"{self._meta}\n"

        for array_name in [
            "data_ref",
            "data_acq",
            "kspace_data",
            "kspace_mask",
            "roi",
            "smaps",
        ]:
            try:
                array = getattr(self, array_name)
            except UndefinedArrayError:
                ret += f"{array_name}: undefined\n"
            else:
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
