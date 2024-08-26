import hydra


from snake_toolkit.cli.acquisition import acquisition
from snake_toolkit.cli.config import ConfigSNAKE
from snake_toolkit.cli.reconstruction import reconstruction


def main(cfg: ConfigSNAKE) -> None:
    """Do Acquisition and Reconstruction sequentially."""
    acquisition(cfg)
    reconstruction(cfg)


main_cli = hydra.main(
    version_base=None, config_path="../../cli-conf/", config_name="config"
)(main)

if __name__ == "__main__":
    main_cli()
