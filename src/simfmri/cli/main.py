#!/usr/bin/env python
"""Main script fot the reconstruction validation."""
import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping
import pickle

import hydra
import numpy as np
from hydra_callbacks import PerfLogger
from omegaconf import DictConfig, OmegaConf

from simfmri.analysis.stats import contrast_zscore, get_scores
from simfmri.handlers import HandlerChain
from simfmri.reconstructors import get_reconstructor
from simfmri.cli.utils import hash_config

logger = logging.getLogger(__name__)


def reconstruct(
    sim_file: os.PathLike, rec_name: str, params: Mapping[str, Any]
) -> tuple[np.ndarray, str]:
    """Reconstruct the data."""
    sim = pickle.load(open(sim_file, "rb"))
    rec = get_reconstructor(rec_name)(**params)
    with PerfLogger(logger, name="Reconstruction " + str(rec)):
        rec.setup(sim)
        return rec.reconstruct(sim), str(rec)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main_app(cfg: DictConfig) -> None:
    """Perform simulation, reconstruction and validation of fMRI data."""
    logger.debug(OmegaConf.to_yaml(cfg))
    logging.captureWarnings(True)

    cache_dir = Path(cfg.cache_dir or os.getcwd())
    hash_sim = hash_config(cfg.simulation, cfg.ignore_patterns)
    sim_file = cache_dir / f"{hash_sim}.pkl"
    # 1. Simulate (use cache if available)
    with PerfLogger(logger, name="Simulation"):
        try:
            if cfg.force_sim:
                raise FileNotFoundError
            sim = pickle.load(open(sim_file, "rb"))
        except OSError:
            logger.warning(f"Failed to load simulation from cache {sim_file}")
            simulator, sim = HandlerChain.from_conf(cfg.simulation)
            sim = simulator(sim)
            del simulator
            os.makedirs(sim_file.parent, exist_ok=True)
            pickle.dump(sim, open(sim_file, "wb"))
    gc.collect()

    reconstructors: Mapping[str, Any] = OmegaConf.to_container(cfg.reconstructors)
    logger.debug("Reconstructors: %s", reconstructors)
    results = []
    # 2. Reconstruct and analyze
    for rec_name, params in reconstructors.items():
        data_test, rec_str = reconstruct(sim_file, rec_name, params)
        logger.debug("Current simulation state: %s", sim)
        with PerfLogger(logger, name="Analysis " + str(rec_str)):
            zscore = contrast_zscore(data_test, sim, cfg.stats.contrast_name)
            stats = get_scores(zscore, sim.roi)

        np.save(f"data_rec_{rec_str}.npy", data_test)
        np.save(f"data_zscore_{rec_str}.npy", zscore)
        results.append(
            {
                "sim_params": OmegaConf.to_container(cfg.simulation.sim_params),
                "handlers": OmegaConf.to_container(cfg.simulation.handlers),
                "reconstructor": rec_str,
                "stats": stats,
                "data_zscore": os.path.join(os.getcwd(), f"data_zscore_{rec_str}.npy"),
                "data_rec": os.path.join(os.getcwd(), f"data_rec_{rec_str}.npy"),
                "sim_file": str(sim_file.absolute()),
            }
        )
        del data_test
        del zscore
        gc.collect()

    # 3. Clean up and saving
    with PerfLogger(logger, name="Cleanup"):
        with open("results.json", "w") as f:
            json.dump(results, f)

    PerfLogger.recap(logger)


if __name__ == "__main__":
    main_app()
