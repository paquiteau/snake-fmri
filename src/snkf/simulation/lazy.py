"""Lazy Simulation Array Module.

very close (i.e. copy pasted and type annotated) to what is done in lazzyarray

https://github.com/NeuralEnsemble/lazyarray/blob/master/lazyarray.py

"""

from __future__ import annotations
import operator
from copy import deepcopy
from typing import Any, Callable, TypeVar, Mapping
import numpy as np
from numpy.typing import ArrayLike, NDArray, DTypeLike
from functools import wraps

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def reverse(func: Callable[[T, U], V]) -> Callable[[U, T], V]:
    """Flip argument of function f(a, b) ->  f(b, a)."""

    @wraps(func)
    def reversed_func(a: U, b: T) -> V:
        return func(b, a)

    reversed_func.__doc__ = "Reversed argument form of %s" % func.__doc__
    reversed_func.__name__ = "reversed %s" % func.__name__
    return reversed_func


def lazy_inplace_operation(name: str) -> LazyOpType:
    """Create a lazy inplace operation on a LazySimArray."""

    def op(self: LazySimArray, val: ArrayLike) -> LazySimArray:
        self.apply(getattr(operator, name), val)
        return self

    return op


def lazy_operation(name: str, reversed: bool = False) -> LazyOpType:
    """Create a lazy operation on a LazySimArray."""

    def op(self: LazySimArray, val: ArrayLike) -> LazySimArray:
        new_map = deepcopy(self)
        f = getattr(operator, name)
        if reversed:
            f = reverse(f)
        new_map.apply(f, val)
        return new_map

    return op


def lazy_unary_operation(name: str) -> Callable[[LazySimArray], LazySimArray]:
    """Create a lazy unary operation on a LazySimArray."""

    def op(self: LazySimArray) -> LazySimArray:
        new_map = deepcopy(self)
        new_map._operations.append((getattr(operator, name), None, None))
        return new_map

    return op


class LazySimArray:
    """A lazy array for the simulation of the data.

    The simulation data is acquired frame wise. The idea is thus to register all
    the required operation to produce this frame.

    This is very close to what is done in larray[1]_ libray, but evaluation will
    alwaysbe considered frame wise.

    .. [1] https://github.com/NeuralEnsemble/lazyarray/tree/master

    """

    def __init__(
        self,
        base_array: NDArray | LazySimArray,
        n_frames: int = -1,
    ):
        self._n_frames = n_frames
        if isinstance(base_array, LazySimArray) and n_frames is None:
            self._n_frames = len(base_array)
        self._base_array = base_array
        self._operations: list[tuple] = []

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape."""
        return (len(self), *(self._base_array.shape))

    @property
    def dtype(self) -> DTypeLike:
        """Get dtype."""
        return self._base_array.dtype

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return len(self.shape) + 1

    def copy(self) -> LazySimArray:
        """Get a copy."""
        return deepcopy(self)

    def __len__(self) -> int:
        """Get length."""
        return self._n_frames

    def __getitem__(self, addr: int | tuple[slice | int]) -> NDArray:
        """Get frame idx by applying all the operations in order.

        If an operation requires the frame index, (ie has `frame_idx=None` in signature)
        it will be provided here.

        """
        match addr:
            case int():
                return self._get_frame(addr)
            case (int() as frame_idx, slicer):
                return self._get_frame(frame_idx)[slicer]
            case (slice(start=start, stop=stop, step=step), slicer):
                start = start or 0
                stop = stop or len(self)
                step = step or 1
                return np.concatenate(
                    [
                        self._get_frame(i)[np.newaxis, slicer]
                        for i in range(start, stop, step)
                    ]
                )
            case _:
                raise ValueError("Index should be a int or tuple of int and slice")

    def _get_frame(self, frame_idx: int) -> NDArray:
        if isinstance(self._base_array, LazySimArray):
            cur = self._base_array[frame_idx]
        else:
            cur = self._base_array
        for op, args, kwargs in self._operations:
            code = getattr(op, "__code__", None)
            if code and "frame_idx" in code.co_varnames[: code.co_argcount]:
                kwargs["frame_idx"] = frame_idx
            cur = op(cur, *args, **kwargs)

        return cur

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def apply(
        self,
        op: Callable[..., NDArray],
        *args: Any,
        **op_kwargs: Mapping[str, Any],
    ) -> None:
        """Register an operation to apply."""
        self._operations.append((op, args, op_kwargs))

    def pop_op(self, idx: int) -> tuple:
        """Pop an operation."""
        op = self._operations.pop(idx)
        return op

    # define standard operations
    __iadd__ = lazy_inplace_operation("add")
    __isub__ = lazy_inplace_operation("sub")
    __imul__ = lazy_inplace_operation("mul")
    __idiv__ = lazy_inplace_operation("div")
    __ipow__ = lazy_inplace_operation("pow")

    __add__ = lazy_operation("add")
    __radd__ = __add__
    __sub__ = lazy_operation("sub")
    __rsub__ = lazy_operation("sub", reversed=True)
    __mul__ = lazy_operation("mul")
    __rmul__ = __mul__
    __div__ = lazy_operation("div")
    __rdiv__ = lazy_operation("div", reversed=True)
    __truediv__ = lazy_operation("truediv")
    __rtruediv__ = lazy_operation("truediv", reversed=True)
    __pow__ = lazy_operation("pow")

    __neg__ = lazy_unary_operation("neg")
    __pos__ = lazy_unary_operation("pos")
    __abs__ = lazy_unary_operation("abs")


LazyOpType = Callable[[LazySimArray, ArrayLike], LazySimArray]
