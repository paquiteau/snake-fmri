#!/usr/bin/env python3
"""Script to create a base simulation, that can be extended."""

from omegaconf import OmegaConf
from hydra.utils import instantiate
import argparse

from simfmri.simulator.handlers import SaveDataHandler


def main() -> None:
    """Create simulation and save to disk."""
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="scenario preset")
    parser.add_argument("output_file", help="save the simulation data")

    ns = parser.parse_args()

    cfg = OmegaConf.load(ns.scenario)

    factory = instantiate(cfg)

    print(factory.handlers)
    if not isinstance(factory.handlers[-1], SaveDataHandler):
        factory.handlers.append(SaveDataHandler(ns.output_file))

    else:
        factory.handlers[-1].sim_file = ns.output_file

    factory.simulate()


if __name__ == "__main__":
    main()
