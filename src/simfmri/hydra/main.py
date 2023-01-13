"""Main script fot the reconstruction validation."""
import logging

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from simfmri.glm import compute_confusion, compute_stats, compute_test

from .logger import PerfLogger
from .utils import dump_confusion, save_data


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_app(cfg: DictConfig) -> None:
    """Perform simulation, reconstruction and validation of fMRI data."""
    # 1. Setup
    simulation_factory = hydra.utils.instantiate(cfg.simulation)
    reconstructor = hydra.utils.instantiate(cfg.reconstruction)

    log.debug(OmegaConf.to_yaml(cfg))

    # 2. Run
    with PerfLogger(log, name="Simulation"):
        sim = simulation_factory.simulate()

    with PerfLogger(log, name="Reconstruction"):
        data_test = reconstructor.reconstruct(sim)

    if len(sim.shape) == 2:
        # fake the 3rd dimension
        data_test = np.expand_dims(data_test, axis=-1)

    with PerfLogger(log, name="Estimation"):
        estimation, design_matrix, contrast = compute_test(
            sim=sim,
            data_test=data_test.T,
            **cfg.stats,
        )
    # 3. Clean up and saving
    contrast = np.squeeze(contrast)
    estimation = np.squeeze(estimation)
    confusion = compute_confusion(estimation.T, sim.roi)

    sim.extra_infos["contrast"] = contrast
    sim.extra_infos["estimation"] = estimation

    if cfg.save_data:
        save_data(cfg.save_data, sim, log)
    confusion_overriden = dump_confusion(confusion)
    log.info(confusion_overriden)
    log.info(compute_stats(**confusion))


if __name__ == "__main__":
    main_app()
