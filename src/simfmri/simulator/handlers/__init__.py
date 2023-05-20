"""Handler Module.

This module gather all simulator build bricks, call handlers,
that can be chained together, to create a fully capable and tailored fMRI simulator.
"""
from .acquisition import AcquisitionHandler, VDSAcquisitionHandler
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
    TextureAdderHandler,
)

__all__ = [
    "AbstractHandler",
    "AcquisitionHandler",
    "ActivationHandler",
    "BigPhantomGeneratorHandler",
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
    "VDSActivationHandler",
]
