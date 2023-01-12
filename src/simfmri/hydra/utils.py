"""Utility function for hydra scripts."""
from hydra.utils import HydraConfig
import json
import os


def safe_cast(val: str) -> int | float | str:
    """Try to cast a value to a number format."""
    try:
        fval = float(val)
    except ValueError:
        return val
    if int(fval) == fval:
        return int(fval)
    return fval


def dump_confusion(results: dict) -> None:
    """Dump the result of the confusion matrix."""
    new_results = results.copy()
    task_overriden = HydraConfig.get().overrides.task
    for overrided in task_overriden:
        key, val = overrided.split("=")
        # keep only the end name of the parameter.
        key = key.split(".")[-1]
        # cast the value to the correct type:
        new_results[key] = safe_cast(val)
    new_results["directory"] = os.getcwd()
    with open("result.json", "w") as f:
        json.dump(new_results, f)
    return new_results
