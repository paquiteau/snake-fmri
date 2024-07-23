"""CLI for SNAKE."""

import logging
import os

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from hydra_callbacks import PerfLogger
from omegaconf import OmegaConf

from snake.handlers import HandlerList, AbstractHandler
from snake.phantom import Phantom
from snake.simulation import SimConfig
from snake.sampling import BaseSampler
from snake.mrd_utils import make_base_mrd, MRDLoader
from snake.engine import BaseAcquisitionEngine

from .config import ConfigSNAKE, snake_handler_resolver, snake_sampler_resolver

OmegaConf.register_new_resolver("snake.handler", snake_handler_resolver)
OmegaConf.register_new_resolver("snake.sampler", snake_sampler_resolver)
cs = ConfigStore.instance()

cs.store(name="config", node=ConfigSNAKE)
for handler_name, cls in AbstractHandler.__registry__.items():
    cs.store(group="handlers", name=handler_name, node={handler_name: cls})

for sampler, cls in BaseSampler.__registry__.items():
    cs.store(group="sampler", name=sampler, node={sampler: cls})


log = logging.getLogger(__name__)


def acquisition(cfg: ConfigSNAKE) -> None:
    """Main function."""

    sim_conf: SimConfig = OmegaConf.to_object(cfg.sim_conf)
    if cfg.phantom.name == "brainweb":
        phantom = Phantom.from_brainweb(sub_id=cfg.phantom.sub_id, sim_conf=sim_conf)
    else:
        raise ValueError(f"Unknown phantom {cfg.phantom.name}")

    handlers = HandlerList.from_cfg(cfg.handlers)

    sampler = cfg.sampler

    for h in handlers:
        phantom = h.get_static(phantom, sim_conf)

    dynamic_data = [h.get_dynamic(phantom, sim_conf) for h in handlers]

    make_base_mrd(cfg.filename, sampler, phantom, sim_conf, dynamic_data)
    log.info("Initialization done")

    engine_klass = BaseAcquisitionEngine.__registry__[sampler.__engine_name__]
    kwargs = {}
    if engine_klass.__engine_name__ == "EPI3dAcquisitionEngine":
        kwargs["nufft_backend"] = cfg.engine.nufft_backend
    engine = engine_klass(mode=cfg.engine.mode, snr=cfg.engine.snr, **kwargs)

    engine(
        cfg.filename,
        worker_chunk_size=cfg.engine.chunk_size,
        n_workers=cfg.engine.n_jobs,
    )

    log.info("Acquisition done")


def reconstruction(cfg: ConfigSNAKE) -> None:
    """Reconstruction function."""
    log.info("Starting reconstruction")
    MRDLoader(cfg.filename)


@hydra.main(version_base=None, config_path="../../cli-conf", config_name="config")
def main_cli(cfg: ConfigSNAKE) -> None:
    """Main App. Does Acquisition and Reconstruction sequentially."""
    acquisition(cfg)
    reconstruction(cfg)


acquisition_cli = hydra.main(
    version_base=None, config_path="../../cli-conf", config_name="config"
)(acquisition)

reconstruction_cli = hydra.main(
    version_base=None, config_path="../../cli-conf", config_name="config"
)(reconstruction)
