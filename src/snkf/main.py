#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig
import numpy as np


@hydra.main(config_path="config.yaml")
def main(cfg: DictConfig) -> None:
