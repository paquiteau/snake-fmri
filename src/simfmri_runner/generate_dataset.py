#!/usr/bin/env python3
"""Generate a dataset for the simulation.

Each sample of the dataset is configured by a config file in the
conf/simulation directory.
"""

import logging
import hashlib
import os
import pandas as pd
import pickle
import hydra
from hydra.experimental.callback import Callback
from hydra.core.utils import JobReturn


from omegaconf import DictConfig, OmegaConf


from .logger import PerfLogger


log = logging.getLogger(__name__)


class RetrieveDatasetCallback(Callback):
    """Retrieve the generated dataset.

    This callback is used to retrieve the generated dataset from each hydra job, and save it to a
    single folder.
    """

    def __init__(self, dataset_dir: str) -> None:
        """Initialize the callback.

        Args:
            output_dir: The directory where the dataset will be saved.
        """
        self.dataset_dir = dataset_dir
        os.mkdir(dataset_dir)
        self.configs = []

    def on_multirun_start(self, config: DictConfig, **kwargs: None) -> None:
        print(os.getcwd())
        print("MULTIRUN_START")

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: None
    ) -> None:
        self.configs.append(
            [config.simulation | {"filename": os.path.abspath(job_return.return_value)}]
        )

    def _remove_target_key(d):
        if isinstance(d, dict):
            for k in list(d.keys()):
                if k == "_target_":
                    del d[k]
                else:
                    self._remove_target_key(d[k])
        elif isinstance(d, list):
            for i in d:
                self._remove_target_key(i)
        return d

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        os.chdir(self.dataset_dir)

        self.configs = self._remove_target_key(self.configs)
        for c in self.configs:
            # Move the file to the current directory
            try:
                os.rename(c["filename"], c["filename"].split("/")[-1])
            except FileExistsError:
                log.info(f"File {c['filename']} already exists, skipping")

        df = pd.DataFrame(self.configs)
        df.to_csv("dataset.csv")


@hydra.main(config_path="conf", config_name="dataset_config")
def generate_data(cfg: DictConfig) -> None:
    """Generate a dataset for the simulation."""
    log.info("Generating dataset")

    if cfg.dry_mode:
        print(OmegaConf.to_yaml(cfg))
        return None

    with PerfLogger(log, name="Simulation"):
        simulation_factory = hydra.utils.instantiate(cfg.simulation)
        sim = simulation_factory.simulate()
        print(sim)
    # Save the dataset
    with PerfLogger(log, name="Saving dataset"):
        filename = hashlib.sha256(pickle.dumps(sim)).hexdigest()
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(sim, f)
        log.info(f"{filename}.pkl")

    return filename


if __name__ == "__main__":
    generate_data()
