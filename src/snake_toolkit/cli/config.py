"""Configuration of SNAKE using Hydra."""

from typing import Any
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from snake.simulation import SimConfig

from snake.handlers import AbstractHandler
from snake.sampling import BaseSampler


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

    name: str = "brainweb"
    sub_id: int = 4
    tissue_select: list[str] = field(default_factory=list)


@dataclass
class ConfigSNAKE:
    """Configuration for SNAKE."""

    handlers: Any
    sampler: Any
    sim_conf: SimConfig = SimConfig()
    engine: EngineConfig = EngineConfig()
    phantom: PhantomConfig = PhantomConfig()

    cache_dir: str = "${oc.env:PWD}/cache"
    result_dir: str = "${oc.env:PWD}/results"
    ignore_patterns: list[str] = field(default_factory=lambda: ["n_jobs"])
    filename: str = "test.mrd"


# Custom Resolver for OmegaConf
# allows to do:
#     _target_: {$snake.handler:motion-image}
# instead of hardcoding the path to the class


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


OmegaConf.register_new_resolver("snake.handler", snake_handler_resolver)
OmegaConf.register_new_resolver("snake.sampler", snake_sampler_resolver)


cs = ConfigStore.instance()
cs.store(name="base_config", node=ConfigSNAKE)

for handler_name, cls in AbstractHandler.__registry__.items():
    cs.store(group="handlers", name=handler_name, node={handler_name: cls})

for sampler, cls in BaseSampler.__registry__.items():
    cs.store(group="sampler", name=sampler, node={sampler: cls})
