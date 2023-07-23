"""Main script fot the reconstruction validation."""
import logging

import hydra
import numpy as np
from hydra_callbacks import PerfLogger
from omegaconf import DictConfig, OmegaConf

from simfmri.runner.glm import get_all_confusion
from simfmri.runner.utils import dump_confusion, save_data
from simfmri.runner.metric import get_snr, get_ptsnr

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

    if len(sim.shape) == 2:
        # fake the 3rd dimension
        data_test = np.expand_dims(data_test, axis=-1)

    log.debug("Current simulation state: %s", sim)
    with PerfLogger(log, name="Estimation"):
        contrast, thresh_mat = get_all_confusion(data_test, sim, **cfg.stats)

        metrics = {
            "ptsnr_roi": get_ptsnr(data_test, sim.data, sim.roi),
            "ptsnr": get_ptsnr(data_test, sim.data),
            "snr_roi": get_snr(data_test, sim.data, sim.roi),
            "snr": get_snr(data_test, sim.data),
        }
    # 3. Clean up and saving
    with PerfLogger(log, name="Cleanup"):
        sim.extra_infos["contrast"] = contrast
        sim.extra_infos["stats"] = thresh_mat
        sim.extra_infos["metrics"] = metrics
        sim.extra_infos["data_test"] = np.squeeze(data_test)

        if cfg.save and cfg.save.data:
            save_data(cfg.save.data, cfg.save.compress, sim, log)

    log.info(PerfLogger.recap())


if __name__ == "__main__":
    main_app()
