"""This hydra script evaluate the reconstruction performance over the dataset."""
import hashlib
import logging
import json
from pathlib import Path
from dataclasses import dataclass

import hydra
import hydra_plugins
import numpy as np
from hydra.core.config_store import ConfigStore

from hydra.core.plugins import Plugins
from hydra_callbacks import PerfLogger
from omegaconf import DictConfig, OmegaConf
from simfmri.runner.metric import get_ptsnr, get_snr
from simfmri.simulator import SimDataType, SimulationData

from ..sweeper import DatasetSweeper
from ..utils import keyval_fmt, log_kwargs, product_dict, setup_warning_logger

# HACK: We get hydra to see the custom sweeper.
# The sweeper is not a plugin, but we can still register it as one.
# Also, the custom sweeper is added as an attribute to the hydra plugins.
DatasetSweeper.__module__ = "hydra_plugins"
hydra_plugins.DatasetSweeper = DatasetSweeper
Plugins.instance().register(DatasetSweeper)

# Then we also register the configuration for the sweeper.


@dataclass
class DatasetSweeperLauncherConfig:
    """Launcher configuration."""

    _target_: str = "hydra_plugins.DatasetSweeper"
    max_batch_size: int = 1
    samples_per_job: int = 1
    dataset_path: str = "datasets/dataset.csv"


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="dataset_sweeper",
    node=DatasetSweeperLauncherConfig,
)


setup_warning_logger()
log = logging.getLogger(__name__)


def get_metrics(
    data_test: np.ndarray,
    sim: SimDataType,
    stat_confs: list[dict],
) -> dict:
    """
    Get all metrics comparing test and references data.

    Parameters
    ----------
    test: np.ndarray
        Test (reconstructed) data
    sim: Simulation
        Original simulation data.
    stat_confs :
    Returns
    -------
    dict:
        Metricsfor the test data.
    """
    metrics = dict()

    if len(sim.shape) == 2:
        # fake the 3rd dimension
        data_test_clean = np.expand_dims(data_test, axis=-1)
    else:
        data_test_clean = data_test

    # compute qualitative metrics
    # TODO parametrize globally the event name.

    for stat_conf in stat_confs:
        thresh_map, dm, contrast_data = compute_test(
            sim=sim,
            data_test=np.moveaxis(data_test_clean, 0, -1),
            **stat_conf,
        )
        # get TP, FP, TN, FN
        confusion_stats = compute_confusion(thresh_map, sim.roi)
        stat_conf_keys = keyval_fmt(**stat_conf)
        for key, value in confusion_stats.items():
            metrics[f"{stat_conf_keys}_{key}"] = value

    # compute quantitative metrics
    metrics["ptsnr"] = get_ptsnr(data_test, sim.data_ref)
    metrics["snr"] = get_snr(data_test, sim.data_ref)
    extend_roi = np.repeat(sim.roi[None, ...], sim.n_frames, axis=0)
    metrics["ptsnr_roi"] = get_ptsnr(data_test, sim.data_ref, extend_roi)
    metrics["snr_roi"] = get_snr(data_test, sim.data_ref, extend_roi)

    return metrics


def do_one_recon(
    sim_file: str, recon_cfg: DictConfig, stat_confs: list[dict]
) -> (np.ndarray, dict):
    """Do one reconstruction using one simulation file."""
    hash_recon = hashlib.sha256(str(recon_cfg).encode("utf-8")).hexdigest()
    log.info(f"Reconstruction method hash: {hash_recon}")
    log.info(f"Reconstruction method: {recon_cfg}")

    with PerfLogger(log, name="Loading data"):
        sim = SimulationData.load_from_file(sim_file, "float32")

    with PerfLogger(log, name="Reconstruction"):
        reconstructor = hydra.utils.instantiate(recon_cfg)
        data_test = reconstructor.reconstruct(sim)

    with PerfLogger(log, name="Metrics computation"):
        metrics = get_metrics(data_test, sim, stat_confs)

    # Save the reconstructed data.
    out_file = f"{Path(sim_file).stem}_{hash_recon}.npy"
    log.info(f"Saving reconstructed data to {out_file}")
    np.save(out_file, data_test)
    metrics |= {
        "sim_file": sim_file,
        "recon_method": hash_recon,
        "recon_method_conf": OmegaConf.to_container(recon_cfg),
    }
    return metrics


@hydra.main(config_path="conf", config_name="eval_recon", version_base=None)
def eval_recon(cfg: DictConfig) -> None:
    """Eval a parametrized reconstruction method on the provided dataset."""
    log.info("Starting reconstruction evaluation")

    if cfg.dry_mode:
        print(OmegaConf.to_yaml(cfg))
        return None

    if isinstance(cfg.dataset_sample, str):
        dataset_samples = [cfg.dataset_sample]
    else:
        dataset_samples = OmegaConf.to_container(cfg.dataset_sample)

    log.info(f"Dataset samples: {type(dataset_samples)}")
    stat_confs = product_dict(**OmegaConf.to_container(cfg.stats))
    results = []
    for filename in dataset_samples:
        metrics = do_one_recon(filename, cfg.reconstruction, stat_confs)
        results.append(metrics)
        log_kwargs(log, **metrics)

    json.dump(results, open("results.json", "w"))
    log.info("Done")


if __name__ == "__main__":
    eval_recon()
