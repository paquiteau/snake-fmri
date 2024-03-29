"""Base Class for Snake-fMRI."""

from __future__ import annotations
import logging
import numpy as np
from numpy.typing import DTypeLike
from typing import Any
from enum import EnumMeta, Enum
import dataclasses
from typing_extensions import dataclass_transform

sim_logger = logging.getLogger("simulation")


####################
# MetaClass Realm  #
####################


@dataclass_transform(kw_only_default=True)
class MetaDCRegister(type):
    """A MetaClass adding registration for subclasses and transform to dataclass."""

    def __new__(
        meta: type[MetaDCRegister],
        clsname: str,
        bases: tuple,
        class_dict: dict,
    ) -> type:
        """Create Handler Class as a dataclass, and register it.

        No need for @dataclass decorator
        """
        cls = dataclasses.dataclass(kw_only=True)(
            super().__new__(meta, clsname, bases, class_dict)  # type: ignore
        )

        name_lookup = f"__{meta.dunder_name}_name__"
        if getattr(cls, "__registry__", None) is None:
            # Adding a registry to the Base Class.
            cls.__registry__ = {}
        if (r := getattr(cls, name_lookup, None)) is not None:
            cls.__registry__[r] = cls
        return cls


class NoCaseEnumMeta(EnumMeta):
    """Make Enum case insensitive."""

    def __getitem__(cls, item: Any):
        if isinstance(item, str):
            item = item.upper()
        return super().__getitem__(item)


class NoCaseEnum(Enum, metaclass=EnumMeta):
    """Base Class for Enum to be case insensitive."""

    pass


################
# Typing Stuff #
################

RngType = int | np.random.Generator | None
"""Type characterising a random number generator.

A random generator is either reprensented by its seed (int),
or a numpy.random.Generator.
"""

AnyShape = tuple[int, ...]


def validate_rng(rng: RngType = None) -> np.random.Generator:
    """Validate Random Number Generator."""
    if isinstance(rng, (int, list)):  # TODO use pattern matching
        return np.random.default_rng(rng)
    elif rng is None:
        return np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        return rng
    else:
        raise ValueError("rng shoud be a numpy Generator, None or an integer seed.")


def cplx_type(dtype: DTypeLike) -> DTypeLike:
    """Return the complex dtype with the same precision as a real one.

    Examples
    --------
    >>> cplx_type(np.float32)
    np.complex64
    """
    d = np.dtype(dtype)
    if d.type is np.complex64:
        return np.complex64
    elif d.type is np.complex128:
        return np.complex128
    elif d.type is np.float64:
        return np.complex128
    elif d.type is np.float32:
        return np.complex64
    else:
        sim_logger.warning(
            f"unsupported dtype {d}, using default complex64", stack_info=True
        )
        return np.complex64


def real_type(
    dtype: DTypeLike,
) -> np.dtype[np.float32] | np.dtype[np.float64]:
    """Return the real type associated with the complex one.

    Examples
    --------
    >>> cplx_type(np.float32)
    np.complex64
    """
    d = np.dtype(dtype)
    if d.type is np.complex64:
        return np.dtype("float32")
    elif d.type is np.float32:
        return np.dtype("float32")
    elif d.type is np.complex128:
        return np.dtype("float64")
    elif d.type is np.float64:
        return np.dtype("float64")
    else:
        sim_logger.warning(
            f"unsupported dtype ({d}) using default float32", stack_info=True
        )
        return np.dtype("float32")


class DuplicateFilter(logging.Filter):
    """
    Filters away duplicate log messages.

    https://stackoverflow.com/a/60462619
    """

    def __init__(self, logger: logging.Logger):
        self.msgs: set[str] = set()
        self.logger = logger

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter duplicate records."""
        msg = str(record.msg)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):
        self.logger.addFilter(self)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        self.logger.removeFilter(self)
