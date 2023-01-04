"""Main script fot the reconstruction validation."""

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_app(cfg: DictConfig) -> None:
    """Perform simulation, reconstruction and validation of fMRI data."""
    print(OmegaConf.to_yaml(cfg, resolve=True))

    simulation_factory = hydra.utils.instantiate(cfg.simulation)
    reconstructor = hydra.utils.instantiate(cfg.reconstruction)

    sim = simulation_factory.simulate()
    np.save("data_ref.npy", sim.data_ref)
    print(reconstructor)


if __name__ == "__main__":
    main_app()
