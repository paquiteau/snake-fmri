"""A dataset sweeper for simfmri_runner."""
import os
from pathlib import Path
import logging
import itertools
from typing import Any, List, Optional, Sequence, Iterable

import pandas as pd


from hydra.types import HydraContext
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.plugins.launcher import Launcher
from hydra.plugins.sweeper import Sweeper
from hydra.types import TaskFunction
from omegaconf import DictConfig, OmegaConf


log = logging.getLogger(__name__)


def chunks(lst: Sequence, n: Optional[int]) -> Iterable[Sequence]:
    """
    Split input to chunks of up to n items each
    """
    if n is None or n == -1:
        n = len(lst)
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class DatasetSweeper(Sweeper):
    """Sweeper loading samples of dataset.

    Parameters
    ----------
    max_batch_size : int
        Maximum batch size.
    dataset_path : str
        Path to dataset.
    """

    def __init__(self, max_batch_size: int, samples_per_job: int, dataset_path: str):
        """Initialize sweeper."""
        self.max_batch_size = max_batch_size
        self.samples_per_job = samples_per_job
        self.dataset_path = dataset_path
        self.dataset_dir = os.path.dirname(self.dataset_path)

        self.config: Optional[DictConfig] = None
        self.launcher: Optional[Launcher] = None
        self.hydra_context: Optional[HydraContext] = None
        self.job_results = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context, task_function=task_function, config=config
        )
        self.hydra_context = hydra_context

        dataset = pd.read_csv(self.dataset_path)
        samples_list = list(dataset["filename"])
        self.samples_list = [
            os.path.abspath(os.path.join(self.dataset_dir, fn)) for fn in samples_list
        ]

        log.info(hydra_context)
        log.info(self.launcher)

    def sweep(self, arguments: List[str]) -> Any:
        assert self.config is not None
        assert self.launcher is not None
        log.info(f"DatasetSweeper(dataset={self.dataset_path}) sweeping")
        log.info(f"Sweep output dir : {self.config.hydra.sweep.dir}")

        # Save sweep run config in top level sweep working directory
        sweep_dir = Path(self.config.hydra.sweep.dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, sweep_dir / "multirun.yaml")

        parser = OverridesParser.create()
        parsed = parser.parse_overrides(arguments)
        lists = []
        # command line provided overrides
        for override in parsed:
            if override.is_sweep_override():
                # Sweepers must manipulate only overrides that return true to is_sweep_override()
                # This syntax is shared across all sweepers, so it may limiting.
                # Sweeper must respect this though: failing to do so will cause all sorts of hard to debug issues.
                # If you would like to propose an extension to the grammar (enabling new types of sweep overrides)
                # Please file an issue and describe the use case and the proposed syntax.
                # Be aware that syntax extensions are potentially breaking compatibility for existing users and the
                # use case will be scrutinized heavily before the syntax is changed.
                sweep_choices = override.sweep_string_iterator()
                key = override.get_key_element()
                sweep = [f"{key}={val}" for val in sweep_choices]
                lists.append(sweep)
            else:
                key = override.get_key_element()
                value = override.get_value_element_as_str()
                lists.append([f"{key}={value}"])
        # Add the override for the dataset
        samples_chunks = []
        for samples in chunks(self.samples_list, self.samples_per_job):
            samples_chunks.append(f"dataset_sample={samples}")
        lists.append(samples_chunks)
        batches = list(itertools.product(*lists))
        print(batches)
        # some sweepers will launch multiple batches.
        # for such sweepers, it is important that they pass the proper initial_job_idx when launching
        # each batch. see example below.
        # This is required to ensure that working that the job gets a unique job id
        # (which in turn can be used for other things, like the working directory)
        chunked_batches = list(chunks(batches, self.max_batch_size))

        returns = []
        initial_job_idx = 0
        for batch in chunked_batches:
            self.validate_batch_is_legal(batch)
            results = self.launcher.launch(batch, initial_job_idx=initial_job_idx)
            initial_job_idx += len(batch)
            returns.append(results)
        return returns
