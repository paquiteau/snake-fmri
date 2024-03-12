#!/usr/bin/env python
"""Main script fot the reconstruction validation."""
import gc
import json
import logging
import os
import dataclasses
from pathlib import Path
from typing import Any, Mapping
import pickle
from pprint import pprint
import numpy as np

import hydra
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from hydra_callbacks import PerfLogger
from omegaconf import OmegaConf

from snkf.analysis.stats import contrast_zscore, get_scores
from snkf.cli.utils import hash_config, snkf_handler_resolver
from snkf.config import ConfigSnakeFMRI

from snkf.handlers import H, HandlerChain
from snkf.reconstructors.base import BaseReconstructor, get_reconstructor

OmegaConf.register_new_resolver("snkf.handler", snkf_handler_resolver)


cs = ConfigStore.instance()

cs.store(name="config", node=ConfigSnakeFMRI)
# add all handlers to the config group
for handler_name, cls in H.items():
    cs.store(group="handlers", name=handler_name, node={handler_name: cls})

# add all handlers to the config group
for reconstructor_name, cls in BaseReconstructor.__registry__.items():
    cs.store(
        group="reconstructors", name=reconstructor_name, node={reconstructor_name: cls}
    )

logger = logging.getLogger(__name__)


def reconstruct(
    sim_file: os.PathLike, rec_name: str, params: Mapping[str, Any]
) -> np.ndarray:
    """Reconstruct the data."""
    sim = pickle.load(open(sim_file, "rb"))
    rec = get_reconstructor(rec_name)(**params)
    with PerfLogger(logger, name="Reconstruction " + str(rec)):
        rec.setup(sim)
        return rec.reconstruct(sim)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main_app(cfg: ConfigSnakeFMRI) -> None:
    """Perform simulation, reconstruction and validation of fMRI data."""
    logging.captureWarnings(True)
    OmegaConf.resolve(cfg)
    pprint(cfg)
    cache_dir = Path(cfg.cache_dir or os.getcwd())
    hash_sim = hash_config(
        dict(
            sim_params=OmegaConf.to_container(cfg.sim_params),
            handlers=OmegaConf.to_container(cfg.handlers),
        ),
        *getattr(cfg, "ignore_patterns", []),
    )
    sim_file = cache_dir / f"{hash_sim}.pkl"
    logger.debug(f"simulation cache file is {sim_file}")
    # 1. Simulate (use cache if available)
    with PerfLogger(logger, name="Simulation"):
        try:
            if cfg.force_sim:
                raise FileNotFoundError
            sim = pickle.load(open(sim_file, "rb"))
        except OSError:
            logger.warning(f"Failed to load simulation from cache {sim_file}")
            handlers = instantiate(cfg.handlers, _convert_="partial")
            simulator, sim = HandlerChain.from_conf(cfg.sim_params, handlers)
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
        if params is None:
            logger.debug(f"Skipped {rec_name}, no parametrization")
            continue
        rec_str = rec_name + hash_config(params)[:5]
        data_test = reconstruct(sim_file, rec_name, params)
        np.save(f"data_rec_{rec_str}.npy", data_test)

        logger.debug("Current simulation state: %s", sim)
        with PerfLogger(logger, name="Analysis " + str(rec_str)):
            zscore = contrast_zscore(data_test, sim, cfg.stats.contrast_name)
            stats = get_scores(zscore, sim.roi)

        np.save(f"data_zscore_{rec_str}.npy", zscore)
        results.append(
            {
                "sim_params": dataclasses.asdict(sim._meta),
                "handlers": OmegaConf.to_container(cfg.handlers),
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
            json.dump(results, f, default=lambda o: str(o))

    PerfLogger.recap(logger)


if __name__ == "__main__":
    main_app()
