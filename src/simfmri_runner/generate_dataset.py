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
        self.configs = []

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: None
    ) -> None:
        self.configs.append(
            OmegaConf.to_container(config.simulation)
            | {"filename": os.path.abspath(job_return.return_value)}
        )

    def _remove_target_key(self, d: dict | list[dict]) -> (dict | list[dict]):
        if isinstance(d, dict):
            # Make a copy of the keys, otherwise the dict complains it has changed size
            for k in list(d.keys()):
                if k == "_target_":
                    del d[k]
                else:
                    self._remove_target_key(d[k])
        elif isinstance(d, list):
            for i in d:
                self._remove_target_key(i)
        return d

    def _unnest_dict(self, d: dict) -> dict:
        """Return a dict where keys are dot-separated."""
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for k2, v2 in self._unnest_dict(v).items():
                    out[f"{k}.{k2}"] = v2
            elif isinstance(v, list):
                out[k] = tuple(v)
            else:
                out[k] = v
        return out

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Save the dataset info to a csv  and copy the dataset to directory."""
        os.chdir(self.dataset_dir)

        print(self.configs)

        self.configs = self._remove_target_key(self.configs)
        self.configs = [self._unnest_dict(c) for c in self.configs]
        print(self.configs)
        for c in self.configs:
            # Move the file to the current directory
            try:
                os.rename(c["filename"], c["filename"].split("/")[-1])
            except FileExistsError:
                log.info(f"File {c['filename']} already exists, skipping")
            # Don't need the full path anymore
            c["filename"] = c["filename"].split("/")[-1]
        df = pd.DataFrame(self.configs)
        try:
            df_orig = pd.read_csv("dataset.csv", index_col=0)
        except FileNotFoundError:
            pass
        else:
            df = pd.concat([df_orig, df], ignore_index=True).drop_duplicates(
                subset="filename", ignore_index=True
            )
        df.to_csv("dataset.csv", mode="w")


@hydra.main(config_path="conf", config_name="dataset_config")
def generate_data(cfg: DictConfig) -> str:
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
        simhash = hashlib.sha256(pickle.dumps(sim)).hexdigest()
        filename = f"{simhash}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(sim, f)
        log.info(filename)

    return filename


if __name__ == "__main__":
    generate_data()
