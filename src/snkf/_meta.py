"""Meta Class Voodoo magic."""

from __future__ import annotations
from typing import Any
from enum import EnumMeta, Enum
import dataclasses
from typing_extensions import dataclass_transform


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
