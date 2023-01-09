# /usr/bin/env python3
"""General utility tools."""

import numpy as np
from time import perf_counter
from logging import Logger
import logging
from typing import Callable


def validate_rng(rng: int | np.random.Generator = None) -> np.random.Generator:
    """Validate Random Number Generator."""
    if isinstance(rng, int):
        return np.random.default_rng(rng)
    elif rng is None:
        return np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        return rng
    else:
        raise ValueError("rng shoud be a numpy Generator, None or an integer seed.")


class PerfLogger:
    """
    Simple Performance logger to use as a context manager.

    Parameters
    ----------
    logger
        A logger object or a callable
    level
        log level, default=0 , infos
    time_format
        format string to display the elapsed time

    Example
    -------
    >>> with PerfLogger(print) as pfl:
            time.sleep(1)
    """

    timers = {}

    def __init__(
        self,
        logger: Logger | Callable,
        level: int = logging.INFO,
        name: str = "default",
        time_format: str = "{name} duration: {:.2f}s",
    ):
        self.logger = logger
        self._name = name
        self._log_level = level
        self._start_time = None
        self._stop_time = None
        self._format = time_format

    def __enter__(self):
        self._start_time = perf_counter()
        return self

    def __exit__(self, *exc_info: tuple):
        self._stop_time = perf_counter()
        elapsed = self._stop_time - self._start_time
        formatted = self._format.format(elapsed, name=self._name)
        self.timers[self._name] = elapsed
        if isinstance(self.logger, Logger):
            self.logger.log(self._log_level, formatted)
        elif callable(self.logger):
            self.logger(formatted)

    def recap(self) -> str:
        """Return a string summarizing all the registered timers."""
        return "\n".join([f"{name}:{t:.2f}s" for name, t in self.timers.items()])
