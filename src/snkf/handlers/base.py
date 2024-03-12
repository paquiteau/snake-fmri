"""Base Handler Interface."""

from __future__ import annotations
import dataclasses
import functools
import time
import copy
import os
import logging
from typing import Callable, Any, Mapping, ClassVar, IO

import yaml
from snkf.config import SimParams
from ..simulation import SimData, UndefinedArrayError
from ..base import MetaDCRegister


CallbackType = Callable[[SimData, SimData], Any]


class MetaHandler(MetaDCRegister):
    """MetaClass for Handlers."""

    dunder_name = "handler"


class AbstractHandler(metaclass=MetaHandler):
    """Handler Interface.

    An Handler is designed to modify a Simulation data object.

    Handlers can be chained using the ``@`` operator.
    Once created, an handler (and its chain of other registered handler) can be applied
    on a simulation using the `handle` function

    Examples
    --------
    >>> A = Handler()
    >>> B = Handler()
    >>> C = Handler() >> A
    >>> s1 = Simulation()
    >>> C.handle(s1.copy()) == B.handle(A.handle(s1))

    """

    __registry__: ClassVar[dict]
    __handler_name__: ClassVar[str]

    def __post_init__(self):
        self._callbacks: list[Callable] = []

    def __rshift__(self, other: AbstractHandler | HandlerChain):
        """Perform self >> other."""
        if isinstance(other, AbstractHandler):
            return HandlerChain(self, other)
        if isinstance(other, SimData):
            return self.handle(other)
        else:
            return NotImplemented

    def __lshift__(self, other: AbstractHandler | HandlerChain):
        """Perform self << other."""
        if isinstance(other, AbstractHandler):
            return HandlerChain(other, self)
        if isinstance(other, SimData):
            return self.handle(other)
        return NotImplemented

    def __call__(self, sim: SimData) -> SimData:
        """Short-hand for handle operation."""
        return self.handle(sim)

    def to_yaml(self) -> str:
        """Show the yaml config associated with the handler."""
        return yaml.dump(dataclasses.asdict(self))

    def __str__(self) -> str:
        ret = ""
        for k, v in dataclasses.asdict(self).items():
            ret += f"{k}={v},"
        name = getattr(self, "name", self.__class__.__name__)
        ret = f"H[{name}]({ret})"
        return ret

    def _run_callbacks(self, old_sim: SimData, new_sim: SimData) -> None:
        """Run the different callbacks.

        Parameters
        ----------
        old_sim
            The simulation object before handling it
        new_sim
            The simulation obkect after handling it
        """
        if isinstance(self._callbacks, list):
            for callback_fun in self._callbacks:
                callback_fun(old_sim, new_sim)
        else:
            try:
                self._callbacks(old_sim, new_sim)
            except TypeError as e:
                raise RuntimeError("callback function not callable.") from e

    @property
    def callbacks(self) -> list[CallbackType]:
        """Return the list of callbacks run after the handling."""
        return self._callbacks

    def add_callback(self, call: CallbackType) -> None:
        """Add a callback to the callback list.

        Parameters
        ----------
        call
            The callback to add
        """
        if not callable(call):
            raise TypeError("Callback attribute should be callable with two argument.")
        if not isinstance(self._callbacks, list):
            self._callbacks = [self._callbacks]

        self._callbacks.append(call)

    def remove_callback(self, idx: int) -> CallbackType:
        """Remove callback according to its position.

        Parameters
        ----------
        idx
            the index of the callback to remove

        Returns
        -------
        callable
            the removed callback.
        """
        return self._callbacks.pop(idx)

    def handle(self, sim: SimData) -> SimData:
        """Handle a specific action done on the simulation, and move to the next one."""
        if self._callbacks:
            old_sim = copy.deepcopy(sim)

        self.log.debug("start handling")
        tic = time.perf_counter()
        new_sim = self._handle(sim)
        toc = time.perf_counter()
        self.log.debug(f"end handling: {toc-tic:.2f}s")

        if self._callbacks:
            self._run_callbacks(old_sim, new_sim)
        return new_sim

    @property
    def log(self) -> logging.Logger:
        """Get a logger."""
        return logging.getLogger(f"simulation.handlers.{self.__class__.__name__}")

    def _handle(self, sim: SimData) -> SimData:
        raise NotImplementedError


class DummyHandler(AbstractHandler):
    """A Handler that does nothing."""

    __handler_name__ = "identity"

    def __init__(self):
        pass

    def _handle(self, sim: SimData) -> SimData:
        return sim


class HandlerChain:
    """Represent a Chain of Handler, that needs to be apply to a simulation."""

    def __init__(self, *args: AbstractHandler):
        self._handlers = list(args)

    def __lshift__(
        self, other: AbstractHandler | HandlerChain | SimData
    ) -> HandlerChain:
        """
        Perform self << other.

        If other is a handler: add it to the beginning of the chain.
        If other is a handler chain: Create a new chain with other chain before self.
        Else: raise NotImplementedError
        If other is a simulation: Apply the chain of handler from left to right
        """
        if isinstance(other, AbstractHandler):
            self._handlers.insert(0, other)
            return self
        elif isinstance(other, HandlerChain):
            return HandlerChain(*other._handlers, *self._handlers)
        elif isinstance(other, SimData):
            return self.__call__(other)
        return NotImplemented

    def __rlshift__(self, other: AbstractHandler) -> HandlerChain:
        """
        Perform other << self.

        If other is handler: create new chain with handler at the end
        """
        if isinstance(other, AbstractHandler):
            self._handlers.append(other)
            return self
        return NotImplemented

    def __rshift__(self, other: AbstractHandler | HandlerChain) -> HandlerChain:
        """
        Perform self >> other.

        If other is a handler, add other to the end of the chain.
        If other is a handler chain: Create a new chain with self before other
        Else: raise NotImplementedError
        """
        if isinstance(other, AbstractHandler):
            self._handlers.append(other)
            return self
        elif isinstance(other, HandlerChain):
            return HandlerChain(*self._handlers, *other._handlers)
        return NotImplemented

    def __rrshift__(
        self, other: AbstractHandler | HandlerChain | SimData
    ) -> HandlerChain:
        """
        Perform other >> self.

        If other is a simulation: run the chain
        else: raise Value Error
        """
        if isinstance(other, AbstractHandler):
            self._handlers.insert(0, other)
            return self
        elif isinstance(other, HandlerChain):
            return HandlerChain(*other._handlers, *self._handlers)
        elif isinstance(other, SimData):
            return self.__call__(other)
        return NotImplemented

    def __eq__(self, other: Any):
        if not isinstance(other, HandlerChain):
            return NotImplemented
        return self._handlers == other._handlers

    def __call__(self, sim: SimData):
        """Apply the handler chain to the simulation.

        We first check if there is a cycling dependency.
        """
        id_list = []
        for h in self._handlers:
            if hid := id(h) not in id_list:
                id_list.append(hid)
            else:
                raise RuntimeError("Cycling list of operator detected.")

        old_sim = sim
        for h in self._handlers:
            new_sim = h.handle(old_sim)
            old_sim = new_sim

        return new_sim

    def add_callback_to_all(self, callback: CallbackType) -> None:
        """Add the same callback to all the handlers."""
        if not callable(callback):
            raise ValueError("Callback should be callable with two arguments.")

        for h in self._handlers:
            h.add_callback(callback)

    def __repr__(self):
        """Represent a simulation."""
        ret = "Handler Chain: "
        for h in self._handlers:
            ret += f"{h} >> "
        ret = ret[:-3]
        return ret

    def to_yaml(self, filename: os.PathLike, sim: SimData | SimParams) -> None | str:
        """Convert a Chain of handler to a yaml representation."""
        conf: dict = {
            "sim_params": {
                f.name: getattr(sim, f.name)
                for f in dataclasses.fields(SimParams)
                if f.name != "extra_infos"
            },
            "handlers": [
                {h.__handler_name__: dataclasses.asdict(h) for h in self._handlers}
            ],
        }
        return yaml.dump(conf, filename)  # type: ignore

    @classmethod
    def from_yaml(cls, stream: bytes | IO[bytes]) -> tuple[HandlerChain, SimData]:
        """Convert a yaml config to a chain of handlers."""
        conf = yaml.safe_load(stream)
        return cls.from_conf(conf["sim_params"], conf["handlers"])

    @classmethod
    def from_conf(
        cls, sim_param: Mapping[str, Any], handlers_conf: Mapping[str, Any]
    ) -> tuple[HandlerChain, SimData]:
        """Load a chain of handler from a configuration."""
        sim = SimData(SimParams(**sim_param))
        handlers = []
        for h_name, h_conf in handlers_conf.items():
            if isinstance(h_conf, AbstractHandler):
                h = h_conf
            else:
                h = AbstractHandler.__registry__[h_name](**h_conf)
            handlers.append(h)
        return HandlerChain(*handlers), sim


# short alias
H = AbstractHandler.__registry__
handler = H


def list_handlers() -> list[str]:
    """List all available handlers."""
    return list(H.keys())


def get_handler(name: str) -> type[AbstractHandler]:
    """Get a handler from its name."""
    return H[name]


def requires_field(
    field_name: str,
    factory: Callable[[SimData], Any] | None = None,
) -> Callable[..., type[AbstractHandler]]:
    """Class Decorator for Handlers.

    This decorator will check if field exist in the simulation object before handling.
    If init_factory is available it will use it to create it, otherwise, an error will
    be raised.

    Parameters
    ----------
    cls: original class
    field_name: str
    factory: callable
    """

    def class_wrapper(cls: type[AbstractHandler]) -> type[AbstractHandler]:
        old_handle = cls.handle

        @functools.wraps(old_handle)
        def wrap_handler(self: AbstractHandler, sim: SimData) -> SimData:
            try:
                getattr(sim, field_name)
            except (UndefinedArrayError, AttributeError) as e:
                if type(e).__name__ == "AttributeError":
                    logging.warn("Unknown attribute")
                if callable(factory):
                    setattr(sim, field_name, factory(sim))
                else:
                    msg = (
                        f"'{field_name}' is missing in simulation"
                        f"and no way of computing it provided for handler {self}."
                    )
                    raise ValueError(msg) from e
            return old_handle(self, sim)

        cls.handle = wrap_handler  # type: ignore

        return cls

    return class_wrapper
