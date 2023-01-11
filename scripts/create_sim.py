#!/usr/bin/env python3
"""
Script to create a base simulation, that can be extended.
"""

from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
import argparse

from simfmri.simulator.handlers import SaveDataHandler


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="scenario preset")
    parser.add_argument("output_file", help="save the simulation data")

    ns = parser.parse_args()

    cfg = OmegaConf.load(ns.scenario)

    factory = instantiate(cfg)

    if not isinstance(factory.handler[-1], SaveDataHandler):
        factory.handlers.append()


if __name__ == "__main__":
    main()
