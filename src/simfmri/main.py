"""Main script fot the reconstruction validation."""

import hydra
from omegaconf import DictConfig

import numpy as np

from simfmri.glm import compute_test, compute_stats


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_app(cfg: DictConfig) -> None:
    """Perform simulation, reconstruction and validation of fMRI data."""
    simulation_factory = hydra.utils.instantiate(cfg.simulation)
    reconstructor = hydra.utils.instantiate(cfg.reconstruction)

    sim = simulation_factory.simulate()

    data_test = reconstructor.reconstruct(sim)
    np.save("data_rec.npy", data_test)

    compute_test = hydra.utils.instantiate(cfg.stats)


if __name__ == "__main__":
    main_app()
