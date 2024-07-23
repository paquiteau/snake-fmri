"""Configuration of SNAKE using Hydra."""

from typing import Any
import hydra

from dataclasses import dataclass, field

from snake.simulation import SimConfig


@dataclass
class EngineConfig:
    """Engine configuration for SNAKE."""

    n_jobs: int = 1
    chunk_size: int = 1
    mode: str = "simple"
    snr: float = float("inf")
    nufft_backend: str = "finufft"


@dataclass
class PhantomConfig:
    """PhantomConfig."""

    name = "brainweb"
    sub_id: int = 4


@dataclass
class ConfigSNAKE:
    """Configuration for SNAKE."""

    sim_config: SimConfig
    handlers: Any
    sampler: Any
    engine: EngineConfig
    phantom: PhantomConfig = PhantomConfig()

    force_sim: bool = False
    cache_dir: str = "${oc.env:PWD}/cache"
    result_dir: str = "${oc.env:PWD}/results"
    ignore_pattern: list[str] = field(default_factory=lambda: ["n_jobs"])
    filename: str = "scenario1.mrd"


def snake_handler_resolver(name: str) -> str:
    """Get Custom resolver for OmegaConf to get handler name."""
    from snake.handlers import H

    cls = H[name]
    return cls.__module__ + "." + cls.__name__


def snake_sampler_resolver(name: str) -> str:
    """Get Custom resolver for OmegaConf to get handler name."""
    from snake.sampling import BaseSampler

    cls = BaseSampler.__registry__[name]
    return cls.__module__ + "." + cls.__name__
