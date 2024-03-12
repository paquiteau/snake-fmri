"""Utility function for hydra scripts."""

from typing import Mapping, Any, Iterable

import re
import json
import os
import itertools
from collections.abc import Sequence
import logging
from joblib.hashing import hash as jb_hash
from omegaconf import DictConfig, OmegaConf


def setup_warning_logger() -> None:
    """Set up the logging system to catch warnings and format them."""

    # Better formatting for the warnings
    class CustomHandler(logging.Handler):
        """Custom Handler for Logging."""

        def emit(self, record: logging.LogRecord) -> None:
            """Change the warnings record arguments."""
            if record.name == "py.warnings":
                # This is dirty, A better solution could be to use a custom formatter.
                record.args = (record.args[0].splitlines()[0],)  # type: ignore
            self.format(record)

    # Configure the logger to catch warnings.
    logging.captureWarnings(True)
    warn_logger = logging.getLogger("py.warnings")
    handler = CustomHandler()
    warn_logger.addHandler(handler)


def safe_cast(val: str) -> int | float | str:
    """Try to cast a value to a number format."""
    try:
        fval = float(val)
    except ValueError:
        return val
    if int(fval) == fval:
        return int(fval)
    return fval


def product_dict(**kwargs: Iterable[Any]) -> list[dict]:
    """Generate a list of dict from the cartesian product of dict values.

    References
    ----------
    https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    """
    keys = kwargs.keys()
    vals = [[val] if not isinstance(val, Sequence) else val for val in kwargs.values()]
    return [dict(zip(keys, inst, strict=True)) for inst in itertools.product(*vals)]


def keyval_fmt(**kwargs: None) -> str:
    """Format a dict as a string compactly."""
    ret = ""
    for key, value in kwargs.items():
        if isinstance(value, float):
            ret += f"{key}{value:.2f}_"
        else:
            ret += f"{value}_"
    # remove last _
    ret = ret[:-1]
    return ret


def aggregate_results(results_files: list[str]) -> str:
    """Aggregate all the .json results files into a single parquet file."""
    import pandas as pd

    results = []
    for r in results_files:
        data = json.load(open(r))
        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)

    results_proc = pd.json_normalize(results)
    df = pd.DataFrame(results_proc)
    # Some light preprocessing.
    df.to_parquet("results_gathered.gzip")
    return os.getcwd() + "/results_gathered.gzip"


def flatten(
    d: Mapping[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """
    Flatten a nested dict.

    https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys

    Parameters
    ----------
    d: dict_like, to flatten.

    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, val in enumerate(v):
                items.extend(flatten({str(i): val}, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def hash_config(conf: Mapping[str, Any], *ignore_keys_pattern: str) -> str:
    """Hash the config, while ignoring certains keys."""
    if isinstance(conf, DictConfig):
        conf = OmegaConf.to_container(conf)
    flattened = flatten(conf)
    for k in list(flattened.keys()):
        for key_pattern in ignore_keys_pattern:
            if re.search(key_pattern, k):
                flattened.pop(k)
    print(flattened)
    return jb_hash(flattened)


def snkf_handler_resolver(name: str) -> str:
    """Get Custom resolver for OmegaConf to get handler name."""
    from snkf.handlers import H

    cls = H[name]
    return cls.__module__ + "." + cls.__name__
