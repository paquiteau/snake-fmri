"""Main script fot the reconstruction validation."""
import logging
import json
import os
import hydra
import numpy as np
from hydra_callbacks import PerfLogger
from omegaconf import DictConfig, OmegaConf

from simfmri.runner.stats import contrast_zscore, get_scores
from simfmri.runner.utils import save_data

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
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
        reconstructor.setup(sim)
        data_test = reconstructor.reconstruct(sim)

    log.debug("Current simulation state: %s", sim)
    with PerfLogger(log, name="Estimation"):
        zscore = contrast_zscore(data_test, sim, cfg.stats.contrast_name)
        stats = get_scores(
            zscore,
            sim.roi,
            alphas=cfg.stats.alpha,
            height_control=cfg.stats.height_control,
        )

    # 3. Clean up and saving
    with PerfLogger(log, name="Cleanup"):
        sim.extra_infos["zscore"] = zscore
        sim.extra_infos["stats"] = stats
        sim.extra_infos["data_test"] = np.squeeze(data_test)

        results = {"stats": stats, "config": OmegaConf.to_container(cfg)}
        if cfg.save and cfg.save.data:
            filename = save_data(cfg.save.data, cfg.save.compress, sim, log)
            results["data"] = os.path.join(os.getcwd(), filename)

        with open("results.json", "w") as f:
            json.dump(results, f)

    log.info(PerfLogger.recap())


if __name__ == "__main__":
    main_app()
