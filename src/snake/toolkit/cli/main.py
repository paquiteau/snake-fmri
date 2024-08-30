"""Main entry point for the SNAKE CLI.

Performs Acquisition and Reconstruction sequentially.
"""

from snake.toolkit.cli.acquisition import acquisition
from snake.toolkit.cli.config import ConfigSNAKE, make_hydra_cli
from snake.toolkit.cli.reconstruction import reconstruction


def main(cfg: ConfigSNAKE) -> None:
    """Do Acquisition and Reconstruction sequentially."""
    acquisition(cfg)
    reconstruction(cfg)


main_cli = make_hydra_cli(main)

if __name__ == "__main__":
    main_cli()
