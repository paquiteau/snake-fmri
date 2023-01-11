"""Callback mechanism for hydra  jobs."""
import os
import json
import glob

from hydra.experimental.callback import Callback
from omegaconf import DictConfig
import pandas as pd


class MultiRunGatherer(Callback):
    """Callback to gather job results."""

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Run after all job have ended.

        Will write a DataFrame from all the results. at the run location.
        """
        save_dir = config.hydra.sweep.dir
        os.chdir(save_dir)
        results = []
        for filename in glob.glob("*/result.json"):
            with open(filename) as f:
                results.append(json.load(f))

        df = pd.DataFrame(results)
        df.to_csv("agg_results.csv")
