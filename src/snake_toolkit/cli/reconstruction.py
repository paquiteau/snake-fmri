"""CLI for SNAKE."""

import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from snake.handlers import AbstractHandler
from snake.mrd_utils import MRDLoader, read_mrd_header
from snake.sampling import BaseSampler

from .config import (
    ConfigSNAKE,
    snake_handler_resolver,
    snake_sampler_resolver,
)

OmegaConf.register_new_resolver("snake.handler", snake_handler_resolver)
OmegaConf.register_new_resolver("snake.sampler", snake_sampler_resolver)


cs = ConfigStore.instance()
cs.store(name="base_config", node=ConfigSNAKE)

for handler_name, cls in AbstractHandler.__registry__.items():
    cs.store(group="handlers", name=handler_name, node={handler_name: cls})

for sampler, cls in BaseSampler.__registry__.items():
    cs.store(group="sampler", name=sampler, node={sampler: cls})

log = logging.getLogger(__name__)


def reconstruction(cfg: ConfigSNAKE) -> None:
    """Reconstruction function."""
    log.info("Starting Reconstruction")
    hdr = read_mrd_header(cfg.filename)
    _, version, engine = hdr.systemModel.split("-")
    log.info(f"Data from {version}, using engine {engine}")

    # Get the appropriate loader
    # Extract sim_confg
    # Reconstructor.setup(sim_conf) # initialize operators
    # array = Reconstructor.reconstruct(dataloader, sim_conf)

    # Do the statistical analysis
    # extract the ROI from the phantom
    # cfg.stats_conf:
    #  - roi_tissue_name # same as in the activation handler !
    #  - roi_threshold
    #  -
    # Save the results as a .nii file in the result directory (not in the cache directory)
    #
    #


reconstruction_cli = hydra.main(
    version_base=None, config_path="../../cli-conf", config_name="config"
)(reconstruction)

if __name__ == "__main__":
    reconstruction_cli()
