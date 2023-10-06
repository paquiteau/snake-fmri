"""Base Handler Interface."""
from __future__ import annotations
from dataclasses import fields
import inspect
import functools
import time
import copy
import os
import logging
from abc import ABCMeta, abstractmethod
from typing import Callable, Any, Iterable, Mapping, IO

import yaml

from ..simulation import SimData, SimParams


CallbackType = Callable[[SimData, SimData], Any]


class MetaHandler(ABCMeta):
    """A Metaclass for Handlers.

    This metaclass does 3 things:
    - Register all the handlers that have a ``name`` class attribute
    - Save the call to init ( for conversion to config string ) in a `_init_params`
      attribute.
    - Add a _callback = [] attribute, to store possible callbacks

    """

    registry = {}

    def __new__(meta, name, bases, namespace, **class_dict):  # noqa
        """Create new Handler class."""
        cls = super().__new__(meta, name, bases, namespace, **class_dict)
        cls_init = cls.__init__

        @functools.wraps(cls_init)
        def wrap_init(self, *args, **kwargs):  # noqa
            try:
                input_params = inspect.getcallargs(cls_init, self, *args, **kwargs)
            except TypeError as e:
                # re raising from original call
                cls_init(self, *args, **kwargs)
                raise e
            cls_init(self, *args, **kwargs)
            input_params.pop(list(input_params.keys())[0])
            self._init_params = input_params
            self._callbacks = []

        cls.__init__ = wrap_init

        if handler_name := getattr(cls, "name", None):
            meta.registry[handler_name] = cls

        return cls

    @property
    def log(cls) -> logging.Logger:
        """Get a logger."""
        return logging.getLogger(f"simulation.handlers.{cls.__name__}")


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

    def __rshift__(self, other: AbstractHandler | HandlerChain):
        """Perform self >> other."""
        if isinstance(other, AbstractHandler):
            return HandlerChain(self, other)
        else:
            return NotImplemented

    def __lshift__(self, other: AbstractHandler | HandlerChain):
        """Perform self << other."""
        if isinstance(other, AbstractHandler):
            return HandlerChain(other, self)
        return NotImplemented

    def __call__(self, sim: SimData) -> SimData:
        """Short-hand for handle operation."""
        return self.handle(sim)

    def to_yaml(self) -> str:
        """Show the yaml config associated with the handler."""
        if not self._init_params:
            return self.name
        else:
            return yaml.dump({self.name: self._init_params})

    def __str__(self) -> str:
        ret = ""
        for k, v in self._init_params.items():
            ret += f"{k}={v},"
        ret = f"H[{self.name}]({ret})"
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

    @abstractmethod
    def _handle(self, sim: SimData) -> SimData:
        pass


class DummyHandler(AbstractHandler):
    """A Handler that does nothing."""

    name = "identity"

    def __init__(self):
        pass

    def _handle(self, sim: SimData) -> SimData:
        return sim


class HandlerChain:
    """Represent a Chain of Handler, that needs to be apply to a simulation."""

    def __init__(self, *args: list[AbstractHandler]):
        self._handlers = list(args)

    def __lshift__(self, other: AbstractHandler | HandlerChain | SimData):
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

    def __rlshift__(self, other: AbstractHandler):
        """
        Perform other << self.

        If other is handler: create new chain with handler at the end
        """
        if isinstance(other, AbstractHandler):
            self._handlers.append(other)
            return self
        return NotImplemented

    def __rshift__(self, other: AbstractHandler | HandlerChain):
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

    def __rrshift__(self, other: AbstractHandler | HandlerChain | SimData):
        """
        Perform other >> self.

        If other is a simulation: run the chain
        else: raise Value Error
        """
        if isinstance(other, AbstractHandler):
            self._handlers.insert(0, other)
            return self
        elif isinstance(other, HandlerChain):
            return HandlerChain(*other._handlers, self._handlers)
        elif isinstance(other, SimData):
            return self.__call__(other)
        return NotImplemented

    def __eq__(self, other: HandlerChain):
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

    def to_yaml(
        self, filename: os.PathLike = None, sim: SimData | SimParams = None
    ) -> None | str:
        """Convert a Chain of handler to a yaml representation."""
        conf = dict()
        if sim:
            conf["sim_params"] = {
                f.name: getattr(sim, f.name)
                for f in fields(SimParams)
                if f.name != "extra_infos"
            }
        conf["handlers"] = []
        for h in self._handlers:
            conf["handlers"].append({h.name: h._init_params})

        return yaml.dump(conf, filename)

    @classmethod
    def from_yaml(cls, stream: bytes | IO[bytes]) -> tuple[HandlerChain, SimData]:
        """Convert a yaml config to a chain of handlers."""
        conf = yaml.safe_load(stream)
        sim = None
        if getattr(conf, "sim_params", None):
            sim = SimData(**conf["sim_params"])
        try:
            handlers_conf = conf["handlers"]
        except KeyError as e:
            raise ValueError(
                "A handler config file should have a  `handlers` section "
            ) from e

        handlers = []
        for hconf in handlers_conf:
            name, conf = list(hconf.items())[0]
            handlers.append(MetaHandler.registry[name](**conf))
        return HandlerChain(*handlers), sim


# short alias
H = MetaHandler.registry
handler = H


def list_handlers() -> list[str]:
    """List all available handlers."""
    return list(MetaHandler.registry.keys())


def get_handler(name: str) -> type(AbstractHandler):
    """Get a handler from its name."""
    return MetaHandler.registry[name]
