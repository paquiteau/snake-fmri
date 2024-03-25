"""A Python Package to simulate fMRI data and validate reconstruction methods."""

from .simulation import SimData, SimParams


from .handlers import (
    H,
    handler,
    get_handler,
    list_handlers,
    AbstractHandler,
    load_from_conf,
    load_from_yaml,
)

import snkf.reconstructors as reconstructors

__all__ = [
    "reconstructors",
    "SimData",
    "SimParams",
    "H",
    "handler",
    "get_handler",
    "list_handlers",
    "AbstractHandler",
    "load_from_conf",
    "load_from_yaml",
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
