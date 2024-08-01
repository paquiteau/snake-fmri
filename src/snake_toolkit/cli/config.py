"""Configuration of SNAKE using Hydra."""

from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

from snake.simulation import SimConfig
from snake.phantom.static import TissueFile
from snake.handlers import AbstractHandler
from snake.sampling import BaseSampler

from snake_toolkit.reconstructors import BaseReconstructor


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
    tissue_ignore: list[str] = field(default_factory=list)
    tissue_file: str | TissueFile = TissueFile.tissue_1T5


@dataclass
class StatConfig:
    """Statistical configuration for SNAKE."""

    roi_tissue_name: str = "ROI"
    roi_threshold: float = 0.5
    event_name: str = "block_on"


@dataclass
class ConfigSNAKE:
    """Configuration for SNAKE."""

    handlers: Any
    sampler: Any
    reconstructors: Any
    sim_conf: SimConfig = SimConfig()
    engine: EngineConfig = EngineConfig()
    phantom: PhantomConfig = PhantomConfig()
    stats: StatConfig = StatConfig()
    cache_dir: Path = "${oc.env:PWD}/cache"  # type: ignore
    result_dir: Path = "${oc.env:PWD}/results"  # type: ignore
    filename: Path = "test.mrd"  # type: ignore


def conf_validator(cfg: DictConfig) -> ConfigSNAKE:
    """Validate the simulation configuration."""
    cfg_obj: ConfigSNAKE = OmegaConf.to_object(cfg)

    cfg_obj.sim_conf.fov_mm = tuple(cfg_obj.sim_conf.fov_mm)
    cfg_obj.sim_conf.shape = tuple(cfg_obj.sim_conf.shape)

    return cfg_obj


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

for reconstructor, cls in BaseReconstructor.__registry__.items():
    cs.store(group="reconstructors", name=reconstructor, node={reconstructor: cls})


def cleanup_cuda():
    """Cleanup CUDA."""
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp._default_memory_pool = cp.cuda.MemoryPool()
    cp._default_pinned_memory_pool = cp.cuda.PinnedMemoryPool()
