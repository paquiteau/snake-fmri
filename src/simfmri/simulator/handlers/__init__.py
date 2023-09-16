"""Handler Module.

This module gather all simulator build bricks, call handlers,
that can be chained together, to create a fully capable and tailored fMRI simulator.
"""
from .acquisition import (
    AcquisitionHandler,
    VDSAcquisitionHandler,
    RadialAcquisitionHandler,
    StackedSpiralAcquisitionHandler,
)
from .activation import ActivationHandler
from .base import AbstractHandler
from .loader import LoadDataHandler, SaveDataHandler
from .noise import (
    GaussianNoiseHandler,
    KspaceNoiseHandler,
    NoiseHandler,
    RicianNoiseHandler,
)
from .phantom import (
    SheppLoganGeneratorHandler,
    SlicerHandler,
    BigPhantomGeneratorHandler,
    RoiDefinerHandler,
    BrainwebPhantomHandler,
    TextureAdderHandler,
)

__all__ = [
    "AbstractHandler",
    "AcquisitionHandler",
    "ActivationHandler",
    "BigPhantomGeneratorHandler",
    "BrainwebPhantomHandler",
    "GaussianNoiseHandler",
    "KspaceNoiseHandler",
    "LoadDataHandler",
    "NoiseHandler",
    "RicianNoiseHandler",
    "RoiDefinerHandler",
    "SaveDataHandler",
    "SheppLoganGeneratorHandler",
    "SlicerHandler",
    "TextureAdderHandler",
    "VDSAcquisitionHandler",
    "RadialAcquisitionHandler",
    "StackedSpiralAcquisitionHandler",
]
