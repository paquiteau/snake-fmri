"""SImulation base objects."""

from __future__ import annotations
from typing import NamedTuple


class GreConfig(NamedTuple):
    """Gradient Recall Echo Sequence parameters."""

    TR: float
    TE: float
    FA: float


class HardwareConfig(NamedTuple):
    """Scanner Hardware parameters."""

    gmax: float
    smax: float
    dwell_time_ms: float
    n_coils: int


class SimConfig(NamedTuple):
    """All base configuration of a simulation."""

    max_sim_time: float
    sim_tr_ms: float
    sequence_params: GreConfig
    hardware: HardwareConfig
    has_relaxation: bool = True


default_hardware = HardwareConfig(gmax=40, smax=200, dwell_time_ms=1e-3, n_coils=8)
