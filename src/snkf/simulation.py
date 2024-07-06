"""SImulation base objects."""

from __future__ import annotations

from dataclasses import dataclass, field, InitVar

import numpy as np


@dataclass(frozen=True)
class GreConfig:
    """Gradient Recall Echo Sequence parameters."""

    TR: float
    TE: float
    FA: float


@dataclass(frozen=True)
class HardwareConfig:
    """Scanner Hardware parameters."""

    gmax: float
    smax: float
    dwell_time_ms: float
    n_coils: int
    field: float = 3.0


default_hardware = HardwareConfig(gmax=40, smax=200, dwell_time_ms=1e-3, n_coils=8)


@dataclass(frozen=True)
class SimConfig:
    """All base configuration of a simulation."""

    max_sim_time: float
    sim_tr_ms: float
    seq: GreConfig
    hardware: HardwareConfig = default_hardware
    fov_mm: tuple[float, float, float] = (192.0, 192.0, 128.0)
    shape: tuple[int, int, int] = (192, 192, 128)  # Target reconstruction shape
    has_relaxation: bool = True
    rng_seed: InitVar[int | None] = None
    rng: np.random.Generator = field(init=False)
    tmp_dir: str = "/tmp"

    def __post_init__(self, rng_seed: int | None):
        # To be compatible with frozen dataclass
        super().__setattr__("rng", np.random.default_rng(rng_seed))
