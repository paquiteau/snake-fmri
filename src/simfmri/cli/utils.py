"""Utility function for hydra scripts."""
from typing import Mapping, Any, Iterable

from hydra.utils import HydraConfig
import re
import json
import os
import itertools
import numpy as np
from collections.abc import Sequence
from simfmri.simulation import SimData
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


def dump_confusion(
    results: dict | list[dict], result_file: str | None = "result.json"
) -> list | dict:
    """Dump the result of the confusion matrix into a json file."""
    if isinstance(results, list):
        ret = []
        for r in results:
            ret.append(dump_confusion(r, result_file=None))
        if result_file:
            with open(result_file, "w") as f:
                json.dump(ret, f)
            return ret
    else:
        new_results: dict[str, Any] = results.copy()
    task_overriden = HydraConfig.get().overrides.task
    for overrided in task_overriden:
        key, val = overrided.split("=")
        # keep only the end name of the parameter.
        key = key.split(".")[-1]
        # cast the value to the correct type:
        new_results[key] = safe_cast(val)
    new_results["directory"] = os.getcwd()
    if result_file:
        with open(result_file, "w") as f:
            json.dump(new_results, f)
    return new_results


def save_data(
    data_saved: str | list[str],
    compress: bool,
    sim: SimData,
    log: logging.Logger,
) -> list[str]:
    """Save part of the data of the simulation.

    Parameters
    ----------
    save_data
        list of attributes to save from the simulation.
        2 Preset are availables:
        - "all" saves all  the array of the simulation
        - "results" saves only the results of the simulation
        - "mini" save the results of the simulation and the reference data.
    sim
        The simulation Data
    log
        Logger
    """
    _save_preset = {
        "all": [
            "data_acq",
            "data_ref",
            "data_test",
            "kspace_data",
            "kspace_mask",
            "roi",
            "estimation",
            "contrast",
            "extra_infos",
        ],
        "results": ["data_test", "estimation", "contrast"],
        "mini": ["data_test", "data_ref", "data_acq"],
    }

    if data_saved is True:
        data_saved = "all"
    if isinstance(data_saved, str):
        try:
            to_save = _save_preset[data_saved]
        except KeyError:
            log.warning(
                "save data preset not found, will try to find attribute matching."
            )
            to_save = [data_saved]

    elif isinstance(data_saved, list):
        to_save = data_saved
    else:
        raise ValueError("data_saved must be a str or a list")

    data_dict = {}
    for data_name in to_save:
        try:
            data_dict[data_name] = getattr(sim, data_name)
        except AttributeError:
            try:
                data_dict[data_name] = sim.extra_infos.get(data_name)
            except KeyError:
                log.warn(f"'{data_name}' not found in simulation")
        if np.iscomplexobj(data_dict[data_name]):
            data_dict[data_name + "_abs"] = np.abs(data_dict[data_name])
    if compress:
        np.savez_compressed("data.npz", **data_dict)
        filename = ["data.npz"]
    else:
        for name, arr in data_dict.items():
            np.save(name, arr)
        filename = [f"{name}.npy" for name in data_dict.keys()]
    log.info(f"saved: {to_save}")
    return filename


def log_kwargs(log: logging.Logger, oneline: bool = False, **kwargs: None) -> None:
    """Log the kwargs."""
    if oneline:
        log.info(f"kwargs: {kwargs}")
        return
    for key, val in kwargs.items():
        log.info(f"{key}: {val}")


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
    for k in flattened:
        for key_pattern in ignore_keys_pattern:
            if re.match(key_pattern, k):
                flattened[k] = "ignored"

    return jb_hash(flattened)
