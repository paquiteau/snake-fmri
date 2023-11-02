"""Main script fot the reconstruction validation."""
import json
import gc
import logging
import os
import pickle
from pathlib import Path

import hydra
import numpy as np
from hydra_callbacks import PerfLogger
from joblib.hashing import hash as jb_hash
from omegaconf import DictConfig, OmegaConf

from simfmri.analysis.stats import contrast_zscore, get_scores
from simfmri.handlers import HandlerChain
from simfmri.reconstructors import get_reconstructor

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main_app(cfg: DictConfig) -> None:
    """Perform simulation, reconstruction and validation of fMRI data."""
    logger.debug(OmegaConf.to_yaml(cfg))
    logging.captureWarnings(True)

    cache_dir = Path(cfg.cache_dir or os.getcwd())
    hash_sim = jb_hash(OmegaConf.to_container(cfg.simulation))
    sim_file = cache_dir / f"{hash_sim}.pkl"
    # 1. Simulate (use cache if available)
    with PerfLogger(logger, name="Simulation"):
        try:
            if cfg.force_sim:
                raise FileNotFoundError
            with open(sim_file, "rb") as f:
                logger.info(f"Loading simulation from cache {sim_file}")
                sim = pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"Failed to load simulation from cache {sim_file}")
            simulator, sim = HandlerChain.from_conf(cfg.simulation)
            sim = simulator(sim)
            del simulator
            with open(sim_file, "wb") as f:
                pickle.dump(sim, f)

    gc.collect()
    reconstructors = OmegaConf.to_container(cfg.reconstructors)
    logger.debug("Reconstructors: %s", reconstructors)
    results = []
    # 2. Reconstruct and analyze
    for rec_name, params in reconstructors.items():
        if len(reconstructors) > 1:
            with open(sim_file, "rb") as f:
                sim = pickle.load(f)  # direct pickling to avoid checkings

        rec = get_reconstructor(rec_name)(**params)
        with PerfLogger(logger, name="Reconstruction " + str(rec)):
            rec.setup(sim)
            data_test = rec.reconstruct(sim)

        logger.debug("Current simulation state: %s", sim)
        with PerfLogger(logger, name="Analysis " + str(rec)):
            zscore = contrast_zscore(data_test, sim, cfg.stats.contrast_name)
            stats = get_scores(zscore, sim.roi)

        np.save(f"data_rec_{rec}.npy", data_test)
        np.save(f"data_zscore_{rec}.npy", zscore)
        results.append(
            {
                "sim_params": OmegaConf.to_container(cfg.simulation.sim_params),
                "handlers": OmegaConf.to_container(cfg.simulation.handlers),
                "reconstructor": str(rec),
                "stats": stats,
                "data_zscore": os.path.join(os.getcwd(), f"data_zscore_{rec}.npy"),
                "data_rec": os.path.join(os.getcwd(), f"data_rec_{rec}.npy"),
                "sim_data": os.path.join(os.getcwd(), "simulation.pkl"),
            }
        )
        del sim
        del rec
        del data_test
        del zscore
        gc.collect()

    # 3. Clean up and saving
    with PerfLogger(logger, name="Cleanup"):
        with open("results.json", "w") as f:
            json.dump(results, f)

    logger.info(PerfLogger.recap())


if __name__ == "__main__":
    main_app()
