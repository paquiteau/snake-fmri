"""Main script fot the reconstruction validation."""
import logging
import json
import os
import hydra
import numpy as np
from hydra_callbacks import PerfLogger
from omegaconf import DictConfig, OmegaConf

from simfmri.simulation import SimData
from simfmri.handlers import HandlerChain
from simfmri.reconstructors import RECONSTRUCTORS

from simfmri.analysis.stats import contrast_zscore, get_scores

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main_app(cfg: DictConfig) -> None:
    """Perform simulation, reconstruction and validation of fMRI data."""
    if cfg.dry_mode:
        print(cfg)
        return None

    log.debug(OmegaConf.to_yaml(cfg))
    OmegaConf.resolve(cfg)
    logging.captureWarnings(True)

    # 2. Run
    with PerfLogger(log, name="Simulation"):
        simulator, sim = HandlerChain.from_conf(cfg.simulation)
        sim = simulator(sim)

    reconstructors = hydra.utils.instantiate(cfg.reconstructors)
    if len(reconstructors) > 1:
        sim.save("simulation.pkl")
    results = []
    for reconf in reconstructors:
        name = list(reconf.keys())[0]
        rec = RECONSTRUCTORS[name](**reconf[name])
        if len(reconstructors) > 1:
            sim = SimData.load_from_file("simulation.pkl", np.float32)
        with PerfLogger(log, name="Reconstruction " + str(rec)):
            rec.setup(sim)
            data_test = rec.reconstruct(sim)

        log.debug("Current simulation state: %s", sim)
        with PerfLogger(log, name="Analysis " + str(rec)):
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

    # 3. Clean up and saving
    with PerfLogger(log, name="Cleanup"):
        with open("results.json", "w") as f:
            json.dump(results, f)

    log.info(PerfLogger.recap())


if __name__ == "__main__":
    main_app()
