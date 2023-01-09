"""Main script fot the reconstruction validation."""

import hydra
from hydra.experimental.callback import Callback
from hydra.core.utils import JobReturn
from omegaconf import DictConfig
from nilearn.plotting import plot_design_matrix
import os
import numpy as np
import pandas as pd
from simfmri.glm import compute_test, compute_confusion


class MultiRunGatherer(Callback):
    """Callback to gather job results."""

    def __init__(self) -> None:
        self.results = []

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs) -> None:
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

    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        """Run after all job have ended.

        Will write a DataFrame from all the results. at the run location.
        """
        print(kwargs)
        df = pd.DataFrame(self.results)
        save_dir = config.hydra.sweep.dir
        print(config)
        df.to_csv(os.path.join(save_dir, "results.csv"))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_app(cfg: DictConfig) -> None:
    """Perform simulation, reconstruction and validation of fMRI data."""
    simulation_factory = hydra.utils.instantiate(cfg.simulation)
    reconstructor = hydra.utils.instantiate(cfg.reconstruction)

    sim = simulation_factory.simulate()

    data_test = reconstructor.reconstruct(sim)
    np.save("data_test_abs.npy", abs(data_test))

    if len(sim.shape) == 2:
        # fake the 3rd dimension
        axis = cfg.simulation.handlers.slicer.axis
        data_test = np.expand_dims(data_test, axis=axis + 1)
        # data_test = np.repeat(data_test, 2, axis + 1)
        print(data_test.shape)
        data_test = data_test.T

    estimation, design_matrix, contrast = compute_test(
        sim=sim,
        data_test=data_test,
        **cfg.stats,
    )
    contrast = np.squeeze(contrast)
    estimation = np.squeeze(estimation)
    plot_design_matrix(design_matrix)
    confusion = compute_confusion(estimation.T, sim.roi)

    if cfg.save_data:
        np.save("data_test_abs.npy", abs(data_test))
        np.save("data_test.npy", data_test)
        np.save("data_ref.npy", sim.data_ref)
        np.save("estimation.npy", estimation.T)

    log.info(confusion)
    log.info(compute_stats(**confusion))
    return confusion


if __name__ == "__main__":
    main_app()
