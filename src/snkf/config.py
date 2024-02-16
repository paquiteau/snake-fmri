"""Configuration setup for Snake-FMRI.

It uses hydra's structured config convention.

"""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from snkf.handlers import H
from snkf.reconstructors.base import BaseReconstructor


@dataclass
class SimParamsConf:
    """Global parameters for the simulation."""

    sim_tr: float = MISSING
    sim_time: float = MISSING
    shape: tuple[int, ...] = (-1, -1, -1)
    fov: tuple[int, ...] = (-1, -1, -1)
    n_coils: int = 1
    lazy: bool = True
    rng: int = 19980804


@dataclass
class ConfigSnakeFMRI:
    """Configuration schema for snake-fmri CLI."""

    sim_params: SimParamsConf
    handlers: Any
    reconstructors: Any
    stats: Any
    force_sim: bool = False
    cache_dir: str = "${oc.env:PWD}/cache"
    result_dir: str = "${oc.env:PWD}/results"
    ignore_pattern: list[str] = field(default_factory=lambda: ["n_jobs"])


cs = ConfigStore.instance()

cs.store(name="config", node=ConfigSnakeFMRI)

# add all handlers to the config group
for handler_name, cls in H.items():
    print(handler_name)
    cs.store(group="handlers", name=handler_name, node={handler_name: cls})

# add all handlers to the config group
for reconstructor_name, cls in BaseReconstructor.__registry__.items():
    print(reconstructor_name)
    cs.store(
        group="reconstructors", name=reconstructor_name, node={reconstructor_name: cls}
    )
