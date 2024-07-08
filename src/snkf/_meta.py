"""Meta Class Voodoo magic."""

from __future__ import annotations

import dataclasses
import itertools
import logging
import sys
from collections.abc import Callable, Generator, Iterable
from enum import Enum, EnumMeta
from typing import Any

from typing_extensions import dataclass_transform


def make_log_property(dunder_name: str) -> Callable:
    """Create a property logger."""

    def log(self: Any) -> logging.Logger:
        """Get a logger."""
        return logging.getLogger(f"{dunder_name}.{self.__class__.__name__}")

    return log


@dataclass_transform(kw_only_default=True)
class MetaDCRegister(type):
    """A MetaClass adding registration for subclasses and transform to dataclass."""

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


class NoCaseEnumMeta(EnumMeta):
    """Make Enum case insensitive."""

    def __getitem__(cls, item: Any):
        if isinstance(item, str):
            item = item.upper()
        return super().__getitem__(item)


class NoCaseEnum(Enum, metaclass=EnumMeta):
    """Base Class for Enum to be case insensitive."""

    pass


if sys.version_info <= (3, 12):

    def batched(iterable: Iterable, n: int) -> Generator[Iterable]:
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch

else:
    batched = itertools.batched
