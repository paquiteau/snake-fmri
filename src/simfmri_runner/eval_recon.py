"""This hydra script evaluate the reconstruction performance over the dataset."""
import hashlib
import logging
import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from simfmri.glm import compute_confusion, compute_test
from simfmri.metrics import get_ptsnr, get_snr
from simfmri.simulation import Simulation

from .logger import PerfLogger

log = logging.getLogger(__name__)


def get_metrics(test: np.ndarray, sim: Simulation) -> dict:
    """
    Get all metrics comparing test and references data.

    Parameters
    ----------
    test: np.ndarray
        Test (reconstructed) data
    sim: Simulation
        Original simulation data.

    Returns
    -------
    dict:
        Metricsfor the test data.
    """
    metrics = dict()

    # compute qualitative metrics
    # TODO parametrize globally the event name.
    for control in ["fpr", "fdr"]:
        thresh_map, dm, contrast_data = compute_test(
            sim,
            test,
            "block_on",
            stat_type="t",
            alpha=0.001,
            height_control=control,
        )
        # get TP, FP, TN, FN
        confusion_stats = compute_confusion(thresh_map, sim.roi)

        for key, value in confusion_stats.items():
            metrics[f"{control}_{key}"] = value

    # compute quantitative metrics
    metrics["ptsnr"] = get_ptsnr(test, sim.data_ref)
    metrics["snr"] = get_snr(test, sim.data_ref)

    metrics["ptsnr_roi"] = get_ptsnr(test, sim.data_ref, sim.roi)
    metrics["snr_roi"] = get_snr(test, sim.data_ref, sim.roi)

    return metrics


@hydra.main(config_path="conf", config_name="eval_recon")
def eval_recon(cfg: DictConfig) -> None:
    """Eval a parametrized reconstruction method on the provided dataset."""
    log.info("Starting reconstruction evaluation")

    if cfg.dry_mode:
        print(OmegaConf.to_yaml(cfg))
        return None

    # load dataset infos
    dataset_df = pd.read_csv(cfg.dataset_path)

    results_df = dataset_df.copy()

    for idx, row in dataset_df.iterrows():
        with PerfLogger(log, name="reconstruction"):
            reconstructor = hydra.utils.instantiate(cfg.reconstruction)

            hash_recon = hashlib.sha256(
                str(cfg.reconstruction).encode("utf-8")
            ).hexdigest()
            log.info(f"Reconstruction method hash: {hash_recon}")
            log.info(f"Reconstruction method: {cfg.reconstruction}")

            sim = Simulation.load(os.path.join(cfg.dataset_path, row["filename"]))
            data_test = reconstructor.reconstruct(sim)

            if len(sim.shape) == 2:
                # fake the 3rd dimension
                data_test = np.expand_dims(data_test, axis=-1)

            metrics = get_metrics(data_test, sim)

            for key, value in metrics.items():
                results_df.loc[idx, key] = value
        if cfg.save:
            np.save(
                os.path.join(cfg.save_path, f"{row['filename']}_{hash_recon}.npy"),
                data_test,
            )

    results_df.to_csv(f"{hash_recon}_stats.csv")
