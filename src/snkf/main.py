#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig
import numpy as np

from .simulation import Phantom, SimConfig


def main(cfg: DictConfig) -> None:

    phantom = Phantom()
