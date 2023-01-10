"""Callback mechanism for hydra  jobs."""
import os

from hydra.experimental.callback import Callback
from hydra.core.utils import JobReturn
from omegaconf import DictConfig
import pandas as pd


class MultiRunGatherer(Callback):
    """Callback to gather job results."""

    def __init__(self) -> None:
        self.results = []

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: None
    ) -> None:
        """Run at the end of each job.

        Notes
        -----
        Needs to be defined in hydra config at `hydra/callbacks/multirun_callback`.
        """
        results = {}
        for overrider in job_return.overrides:
            key, val = overrider.split("=")
            results[key] = val
        results |= job_return.return_value
        self.results.append(results)

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Run after all job have ended.

        Will write a DataFrame from all the results. at the run location.
        """
        df = pd.DataFrame(self.results)
        save_dir = config.hydra.sweep.dir
        df.to_csv(os.path.join(save_dir, "results.csv"))
