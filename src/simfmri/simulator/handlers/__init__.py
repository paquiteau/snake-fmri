"""Handler Module.

This module gather all simulator build bricks, call handlers,
that can be chained together, to create a fully capable and tailored fMRI simulator.
"""
from .base import AbstractHandler

from .phantom import SheppLoganGeneratorHandler, SlicerHandler

from .noise import (
    NoiseHandler,
    GaussianNoiseHandler,
    RicianNoiseHandler,
    KspaceNoiseHandler,
)

from .acquisition import AcquisitionHandler
from .activation import ActivationHandler

__all__ = [
    "AbstractHandler",
    "AcquisitionHandler",
    "ActivationHandler",
    "GaussianNoiseHandler",
    "KspaceNoiseHandler",
    "NoiseHandler",
    "RicianNoiseHandler",
    "SheppLoganGeneratorHandler",
    "SlicerHandler",
]
