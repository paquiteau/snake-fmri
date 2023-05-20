"""Main script fot the reconstruction validation."""
import logging

import hydra
import numpy as np
from hydra_callbacks import PerfLogger
from omegaconf import DictConfig, OmegaConf

from simfmri.glm import compute_confusion, compute_confusion_stats, compute_test

from .utils import dump_confusion, product_dict, save_data

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_app(cfg: DictConfig) -> None:
    """Perform simulation, reconstruction and validation of fMRI data."""
    if cfg.dry_mode:
        print(cfg)
        return None

    # 1. Setup
    simulation_factory = hydra.utils.instantiate(cfg.simulation)
    reconstructor = hydra.utils.instantiate(cfg.reconstruction)

    log.debug(OmegaConf.to_yaml(cfg))
    logging.captureWarnings(True)

    # 2. Run
    with PerfLogger(log, name="Simulation"):
        sim = simulation_factory.simulate()

    with PerfLogger(log, name="Reconstruction"):
        data_test = reconstructor.reconstruct(sim)

    if len(sim.shape) == 2:
        # fake the 3rd dimension
        data_test = np.expand_dims(data_test, axis=-1)

    log.debug("Current simulation state: %s", sim)
    with PerfLogger(log, name="Estimation"):
        stat_configs = product_dict(**OmegaConf.to_container(cfg.stats))
        contrasts = []
        estimations = []
        confusions = []
        for stat_conf in stat_configs:
            estimation, design_matrix, contrast = compute_test(
                sim=sim,
                data_test=np.moveaxis(data_test, 0, -1),
                **stat_conf,
            )
            contrasts.append(np.squeeze(contrast))
            estimations.append(np.squeeze(estimation))
            confusion = compute_confusion(np.squeeze(estimation), sim.roi)
            confusion |= stat_conf
            confusions.append(confusion)
    # 3. Clean up and saving
    with PerfLogger(log, name="Cleanup"):
        sim.extra_infos["contrast"] = contrasts
        sim.extra_infos["estimation"] = estimations

        sim.extra_infos["data_test"] = np.squeeze(data_test)

        if cfg.save and cfg.save.data:
            save_data(cfg.save.data, cfg.save.compress, sim, log)
        confusion_overriden = dump_confusion(confusions)
    for c in confusion_overriden:
        log.info(
            c | compute_confusion_stats(c["f_neg"], c["t_neg"], c["f_pos"], c["t_pos"])
        )
    log.info(PerfLogger.recap())


if __name__ == "__main__":
    main_app()
