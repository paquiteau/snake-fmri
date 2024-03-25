"""Configuration setup for Snake-FMRI.

It uses hydra's structured config convention.

"""

from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class HardwareParams:
    """Parameters describing a MRI Scanner."""

    field: int = 3
    """Field Strength."""
    n_coils: int = 1
    """Number of coils in antenna."""
    gmax: float = 0.04
    """Max Gradient Strength (T/m/s)."""
    smax: float = 0.1
    """Max Slew Rate Strength. (T/m/s/s)"""
    raster_time: float = 0.01
    """Gradient Raster time (s)."""


@dataclass
class SimParams:
    """Simulation metadata."""

    shape: tuple[int, ...]
    """Shape of the volume of the simulation."""
    sim_time: float
    """Total Simulation time in seconds."""
    sim_tr: float
    """Time resolution for the simulation."""
    n_coils: int = 1
    """Number of coil of the simulation."""
    rng: int = 19980408
    """Random number generator seed."""
    hardware: HardwareParams = field(default_factory=HardwareParams)
    """Hardware Configuration."""
    extra_infos: dict[str, Any] = field(default_factory=lambda: dict(), repr=False)
    """Extra information, to add more information to the simulation"""
    fov: tuple[float, ...] = (-1, -1, -1)
    """Field of view of the volume in mm"""
    lazy: bool = False
    """Is the computations lazy ?"""

    def __post_init__(self) -> None:
        self.n_frames = int(self.sim_time / self.sim_tr)


@dataclass
class ConfigSnakeFMRI:
    """Configuration schema for snake-fmri CLI."""

    handlers: Any
    reconstructors: Any
    stats: Any
    sim_params: SimParams
    force_sim: bool = False
    cache_dir: str = "${oc.env:PWD}/cache"
    result_dir: str = "${oc.env:PWD}/results"
    ignore_pattern: list[str] = field(default_factory=lambda: ["n_jobs"])
