"""SNAKE-fMRI core package."""

from .simulation import SimConfig, default_hardware, GreConfig, HardwareConfig

from .phantom import Phantom, DynamicData
from .smaps import get_smaps
from .sampling import (
    EPI3dAcquisitionSampler,
    BaseSampler,
    StackOfSpiralSampler,
    NonCartesianAcquisitionSampler,
)

from .engine import BaseAcquisitionEngine, EPIAcquisitionEngine, NufftAcquisitionEngine

__all__ = [
    "BaseAcquisitionEngine",
    "BaseSampler",
    "DynamicData",
    "EPI3dAcquisitionSampler",
    "EPIAcquisitionEngine",
    "GreConfig",
    "HardwareConfig",
    "NonCartesianAcquisitionSampler",
    "NufftAcquisitionEngine",
    "Phantom",
    "SimConfig",
    "StackOfSpiralSampler",
    "default_hardware",
    "get_smaps",
]
