"""Callback mechanism for hydra  jobs."""
import os
import errno
import json
import glob

from hydra.experimental.callback import Callback
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import pandas as pd


class MultiRunGatherer(Callback):
    """Define a callback to gather job results."""

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Run after all job have ended.

        Will write a DataFrame from all the results. at the run location.
        """
        save_dir = config.hydra.sweep.dir
        os.chdir(save_dir)
        results = []
        for filename in glob.glob("*/result.json"):
            with open(filename) as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    results.extend(loaded)
                else:
                    results.append(loaded)
        df = pd.DataFrame(results)
        df.to_csv("agg_results.csv")


class LatestRunLink(Callback):
    """Callback that create a symlink to the latest run in the base output dir.

    Parameters
    ----------
    run_base_dir
        name of the basedir
    multirun_base_dir
    """

    def __init__(
        self, run_base_dir: str = "outputs", multirun_base_dir: str = "multirun"
    ):
        self.run_base_dir = to_absolute_path(run_base_dir)
        self.multirun_base_dir = to_absolute_path(multirun_base_dir)

    def on_run_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute after a single run."""
        self._on_anyrun_end(config.hydra.run.dir, self.run_base_dir)

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute after a multi run."""
        self._on_anyrun_end(config.hydra.sweep.dir, self.multirun_base_dir)

    def _on_anyrun_end(self, run_dir: str, base_dir: str) -> None:
        self._force_symlink(
            to_absolute_path(run_dir),
            to_absolute_path(os.path.join(base_dir, "latest")),
        )

    def _force_symlink(self, src: str, dest: str) -> None:
        """Create a symlink from src to test, overwriting dest if necessary."""
        try:
            os.symlink(src, dest)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(dest)
                os.symlink(src, dest)
            else:
                raise e
