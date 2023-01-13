"""Utility function for hydra scripts."""
from hydra.utils import HydraConfig
import json
import os
import numpy as np
from simfmri.simulator import SimulationData
from logging import Logger


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
    """Dump the result of the confusion matrix into a json file."""
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


def save_data(save_data: str | list[str], sim: SimulationData, log: Logger) -> None:
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
        ],
        "results": ["data_test", "estimation", "contrast"],
        "mini": ["data_test", "data_ref", "data_acq"],
    }

    if isinstance(save_data, str):
        try:
            to_save = _save_preset[save_data]
        except KeyError:
            log.error("save data preset not found.")

    if isinstance(save_data, list):
        to_save = save_data
    data_dict = {}
    for data_name in to_save:
        try:
            data_dict[data_name] = getattr(sim, data_name)
        except AttributeError:
            try:
                data_dict[data_name] = sim.extra_infos.get(data_name)
            except KeyError:
                log.warn(f"'{data_name}' not found in simulation")

    np.savez_compressed("data.npz", data_dict)
    log.info(f"saved: {to_save}")
