"""Handler Module.

This module gather all simulator build bricks, call handlers,
that can be chained together, to create a fully capable and tailored fMRI simulator.
"""
from pathlib import Path
import importlib
import pkgutil

# load all the interfaces modules
for _, name, _ in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if name.startswith("_"):
        continue
    importlib.import_module("." + name, __name__)

from .base import (
    AVAILABLE_HANDLERS,
    H,
    handler,
    get_handler,
    list_handlers,
    AbstractHandler,
)

__all__ = [
    "AVAILABLE_HANDLERS",
    "H",
    "handler",
    "get_handler",
    "list_handlers",
    "AbstractHandler",
]