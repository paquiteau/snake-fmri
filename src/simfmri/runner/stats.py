"""Analysis module."""
import logging
from typing import Literal

from sklearn.metrics import (
    auc,
    accuracy_score,
    jaccard_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np

from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm import compute_contrast, expression_to_contrast_vector
from nilearn.glm.thresholding import fdr_threshold
from scipy.stats import norm

from simfmri.simulator import SimulationData

logger = logging.getLogger(__name__)


def get_contrast_zscore(
    image: np.ndarray,
    sim: SimulationData,
    contrast_name: str,
    **kwargs: None,
) -> tuple[np.ndarray, dict]:
    """Compute z-score of contrast_name.

    For now only a single contrast is supported.

    Parameters
    ----------
    image : np.ndarray
        4D image data.
    sim : SimulationData
        Simulation data object.
    contrast_name : str
        Contrast name.
    alpha : float or list of float, optional
        Alpha level(s) for thresholding, by default 0.001.
    height_control : str, optional
        Height control method, by default "fpr".
    **kwargs : dict
        Additional arguments passed to `nilearn.glm.compute_contrast`.

    Returns
    -------
    z_image : np.ndarray
        Z-score image.
    thresh_dict : dict
        Dictionary of thresholded images for each alpha level.

    """
    design_matrix = make_first_level_design_matrix(
        frame_times=np.arange(len(image)) * sim.extra_infos["TR_ms"] / 1000,
        events=sim.extra_infos["events"],
        drift_model=sim.extra_infos.get("drift_model", None),
    )
    # Create a mask from reference data (not ideal, but best)
    mask = sim.data_ref[0] > 0
    image_ = abs(image)[:, mask]

    labels, results = run_glm(image_, design_matrix.values)
    # Translate formulas to vectors
    con_val = expression_to_contrast_vector(
        contrast_name, design_matrix.columns.tolist()
    )

    contrast = compute_contrast(labels, results, con_val, contrast_type="t")

    z_image = np.zeros(mask.shape)
    z_image[mask] = contrast.z_score()

    return z_image


def get_thresh_map(
    z_image: np.ndarray,
    alpha: float | list[float],
    height_control: Literal["fpr", "fdr"] = "fpr",
) -> dict[float, np.ndarray]:
    """Get thresholded map."""
    thresh_dict = {}
    if not isinstance(alpha, list):
        alphas = [alpha]
    else:
        alphas = alpha
    for a in alphas:
        if height_control == "fpr":
            z_thresh = norm.isf(a)
        elif height_control == "fdr":
            z_thresh = fdr_threshold(z_image, a)

        above_thresh = z_image > z_thresh
        thresh_dict[a] = above_thresh
    return thresh_dict


def get_confusion_map(
    above_thresh: np.ndarray, ground_truth: np.ndarray
) -> dict[str, np.ndarray]:
    """Get confusion map."""
    return {
        "TP": above_thresh & ground_truth,
        "FP": above_thresh & ~ground_truth,
        "FN": ~above_thresh & ground_truth,
        "TN": ~above_thresh & ~ground_truth,
    }


def get_all_confusion(
    data: np.ndarray, sim: SimulationData, **stat_conf: None
) -> tuple(dict[float, np.ndarray], np.ndarray):
    """Get confusion matrix for all alpha levels.

    Parameters
    ----------
    data : np.ndarray
        4D image data.
    sim : SimulationData
        Simulation data object.
    **stat_conf : dict
        Additional arguments passed to `get_contrast_zscore`.

    Returns
    -------
    conf_mats : dict
        Dictionary of confusion matrices for each alpha level.
    """
    z_image = get_contrast_zscore(data, sim, contrast_name=stat_conf["contrast_name"])
    thresh_dict = get_thresh_map(
        z_image, alpha=stat_conf["alpha"], height_control=stat_conf["height_control"]
    )
    conf_mats = {}
    for alpha, z_thresh in thresh_dict.items():
        conf_map = get_confusion_map(z_thresh, sim.roi)
        conf_map = {k: np.sum(v) for k, v in conf_map.items()}
        # Convert to 2x2 matrix [[TP, FP], [FN, TN]]
        conf_mat = np.array(list(conf_map.values())).reshape(2, 2)
        conf_mats[alpha] = conf_mat
    return conf_mats, z_image


def get_auc(conf_mats: dict[float, np.ndarray]) -> float:
    """Get area under curve for a ROC.

    Parameters
    ----------
    conf_mats : dict
        Dictionary of confusion matrices for each alpha level.

    Returns
    -------
    auc : float
        Area under curve.
    """
    # TPR =  TP / TP + FN
    tprs = [v[0, 0] / (v[0, 0] + v[1, 0]) for k, v in conf_mats.items()]
    # FPR = FP / FP + TN
    fprs = [v[0, 1] / (v[0, 1] + v[1, 1]) for k, v in conf_mats.items()]
    # Add 0,0 to start
    tprs.insert(0, 0)
    fprs.insert(0, 0)
    return auc(np.array(fprs), np.array(tprs))


def get_scores(thresh_map: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Get sklearn metrics scores.

    Parameters
    ----------
    thresh_map : np.ndarray
        Thresholded map.
    ground_truth : np.ndarray
        Ground truth map.

    Returns
    -------
    scores : dict
        Dictionary of scores (accuracy, precision, recall, f1, jaccard)

    """

    return {
        "accuracy": accuracy_score(ground_truth, thresh_map),
        "precision": precision_score(ground_truth, thresh_map),
        "recall": recall_score(ground_truth, thresh_map),
        "f1": f1_score(ground_truth, thresh_map),
        "jaccard": jaccard_score(ground_truth, thresh_map),
    }
