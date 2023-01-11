"""Main script fot the reconstruction validation."""
import logging

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from simfmri.glm import compute_confusion, compute_stats, compute_test

from .logger import PerfLogger
from .utils import dump_confusion


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_app(cfg: DictConfig) -> None:
    """Perform simulation, reconstruction and validation of fMRI data."""
    simulation_factory = hydra.utils.instantiate(cfg.simulation)
    reconstructor = hydra.utils.instantiate(cfg.reconstruction)

    log.debug(OmegaConf.to_yaml(cfg))
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
    contrast = np.squeeze(contrast)
    estimation = np.squeeze(estimation)
    confusion = compute_confusion(estimation.T, sim.roi)

    if cfg.save_data:
        np.save("data_test_abs.npy", np.squeeze(abs(data_test)))
        np.save("data_ref.npy", sim.data_ref)
        np.save("data_acq.npy", sim.data_acq)
        np.save("estimation.npy", estimation)
        log.info("saved: data_test, data_ref, data_acq, estimation")

    confusion_overriden = dump_confusion(confusion)
    log.info(confusion_overriden)
    log.info(compute_stats(**confusion))


if __name__ == "__main__":
    main_app()
