import hydra


from snake.toolkit.cli.acquisition import acquisition
from snake.toolkit.cli.config import ConfigSNAKE
from snake.toolkit.cli.reconstruction import reconstruction


def main(cfg: ConfigSNAKE) -> None:
    """Do Acquisition and Reconstruction sequentially."""
    acquisition(cfg)
    reconstruction(cfg)


main_cli = hydra.main(
    version_base=None, config_path="../../cli-conf/", config_name="config"
)(main)

if __name__ == "__main__":
    main_cli()
