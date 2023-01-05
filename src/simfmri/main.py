"""Main script fot the reconstruction validation."""

import hydra
from omegaconf import DictConfig

import numpy as np


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_app(cfg: DictConfig) -> None:
    """Perform simulation, reconstruction and validation of fMRI data."""
    simulation_factory = hydra.utils.instantiate(cfg.simulation)
    reconstructor = hydra.utils.instantiate(cfg.reconstruction)

    sim = simulation_factory.simulate()

    estimation = reconstructor.reconstruct(sim)

    np.save("data_rec.npy", estimation)


if __name__ == "__main__":
    main_app()
