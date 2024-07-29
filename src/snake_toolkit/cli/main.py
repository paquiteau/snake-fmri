import hydra
from omegaconf import OmegaConf
from snake.handlers import HandlerList, AbstractHandler
from snake.phantom import Phantom
from snake.simulation import SimConfig
from snake.sampling import BaseSampler
from snake.mrd_utils import make_base_mrd, MRDLoader
from snake.engine import BaseAcquisitionEngine


from .acquisition import acquisition
from .config import ConfigSNAKE
from .reconstruction import reconstruction


def main(cfg: ConfigSNAKE) -> None:
    """Do Acquisition and Reconstruction sequentially."""
    acquisition(cfg)
    reconstruction(cfg)


main_cli = hydra.main(
    version_base=None, config_path="../../cli-conf/", config_name="config"
)(main)

if __name__ == "__main__":
    main_cli()
