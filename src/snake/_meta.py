"""Meta Class Voodoo magic."""

from __future__ import annotations
import os
import dataclasses
import itertools
import logging
import sys
from pathlib import Path
from collections import defaultdict
from collections.abc import Callable, Iterator, Iterable
from enum import Enum, EnumMeta
from functools import wraps
from typing import Any, TypeVar
from functools import partial

from typing_extensions import dataclass_transform

T = TypeVar("T")


def make_log_property(dunder_name: str) -> Callable:
    """Create a property logger."""

    def log(self: Any) -> logging.Logger:
        """Get a logger."""
        return logging.getLogger(f"{dunder_name}.{self.__class__.__name__}")

    return log


@dataclass_transform(kw_only_default=True)
class MetaDCRegister(type):
    """A MetaClass adding registration for subclasses and transform to dataclass."""

    dunder_name: str

    def __new__(
        meta: type[MetaDCRegister],
        clsname: str,
        bases: tuple,
        class_dict: dict,
    ) -> type:
        """Create a dataclass, with log property and auto registry properties.

        No need for @dataclass decorator.
        """
        class_dict["log"] = property(make_log_property(meta.dunder_name))
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


class LogMixin:
    """A Mixin to add a logger to a class."""

    @property
    def log(self) -> logging.Logger:
        """Logger."""
        return logging.getLogger(f"{self.__class__.__name__}")


class NoCaseEnumMeta(EnumMeta):
    """Make Enum case insensitive."""

    def __getitem__(cls, item: Any):
        if isinstance(item, str):
            item = item.upper()
        return super().__getitem__(item)


class NoCaseEnum(Enum, metaclass=EnumMeta):
    """Base Class for Enum to be case insensitive."""

    pass


class MethodRegister:
    """
    A Decorator to register methods of the same type in dictionnaries.

    Parameters
    ----------
    name: str
        The  register
    """

    registry: dict = defaultdict(dict)

    def __init__(self, register_name: str):
        self.register_name = register_name

    def __call__(
        self,
        method_name: str | Callable,
    ) -> Callable:
        """Register the function in the registry."""

        def decorator(func: Callable[..., T], method_name: str) -> Callable[..., T]:
            self.registry[self.register_name][method_name] = func
            if func.__name__ != method_name:
                func.__name__ += "__" + method_name

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                return func(*args, **kwargs)

            return wrapper

        # allow for direct name.
        if callable(method_name):
            func = method_name
            method_name = func.__name__
            return decorator(func, method_name)
        else:
            return partial(decorator, method_name=method_name)


if sys.version_info <= (3, 12):

    def batched(iterable: Iterable, n: int) -> Iterator[tuple[int]]:
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch

else:
    batched = itertools.batched


class Singleton(type):
    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ENVCONFIG(metaclass=Singleton):
    """Environment Configuration."""

    SNAKE_TMP_DIR = "/tmp"
    SNAKE_HDF5_CHUNK_SIZE = 1024**2
    SNAKE_HDF5_CHUNK_WRITE_SIZE = 4 * 1024**3

    @classmethod
    def __getitem__(cls, key: str) -> Any:

        if key in os.environ:
            return os.environ[key]
        return getattr(cls, key)


EnvConfig = ENVCONFIG()
