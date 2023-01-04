"""Base Handler Interface."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Callable

from ..simulation import SimulationData


class AbstractHandler(ABC):
    """Handler Interface.

    An Handler is designed to modify a Simulation data object.

    Handlers can be chained using the ``@`` operator.
    Once created, an handler (and its chain of other registered handler) can be applied
    on a simulation using the `handle` function

    Examples
    --------
    >>> A = Handler()
    >>> B = Handler()
    >>> C = Handler() @ A
    >>> s1 = Simulation()
    >>> C.handle(s1.copy()) == B.handle(A.handle(s1))

    """

    def __init__(self):
        self._callbacks = []
        self._next = None

    def __matmul__(self, obj: AbstractHandler):
        """Chain the handler with the righhandside.

        Example
        -------
        >>> a, b = Handler(), Handler()
        >>> a @ b is a.set_next(b)
        True
        >>> a @ b is b
        True
        """
        if isinstance(obj, AbstractHandler):
            return self.set_next(obj)
        else:
            raise TypeError

    def __call__(self, sim: SimulationData) -> SimulationData:
        """Short-hand for handle operation."""
        return self.handle(sim)

    def _run_callbacks(self, old_sim: SimulationData, new_sim: SimulationData):
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
    def callbacks(self) -> list:
        """Return the list of callbacks run after the handling."""
        return self._callbacks

    def add_callback(self, call: Callable):
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

    def remove_callback(self, idx: int):
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
        self._callback.pop(idx)

    @property
    def next(self):
        """Next handler in the chain."""
        return self._next

    @next.setter
    def next(self, handler: AbstractHandler) -> AbstractHandler:
        """Set the next handler to call.

        Parameters
        ----------
        handler
            The next handler to call

        Returns
        -------
        handler
            The next handler to call
        """
        if isinstance(handler, AbstractHandler):
            self._next = handler
        return handler

    def get_chain(self):
        """show the chain of actions that would be applyied to a simulation."""
        cur = self
        handler_chain = []
        while cur._next is not None:
            handler_chain.append(cur)
            cur = cur._next
        ret_str = ""
        for h in handler_chain[::-1]:
            ret_str += f"{h.__class__.__name__}" + "->"

        return ret_str

    def handle(self, sim: SimulationData) -> SimulationData:
        """Handle a specific action done on the simulation, and move to the next one."""
        if self._callbacks is not None:
            old_sim = copy.deepcopy(sim)
            new_sim = self._handle(sim)
            self._run_callbacks(old_sim, new_sim)
        else:
            new_sim = self._handle(sim)
        if self._next:
            return self._next.handle(new_sim)
        else:
            return new_sim

    @abstractmethod
    def _handle(self, sim: SimulationData) -> None:
        pass
