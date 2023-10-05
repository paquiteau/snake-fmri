"""Analysis module."""
import logging
from typing import Literal

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
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

from simfmri.simulation import SimData

logger = logging.getLogger(__name__)

HeightControl = Literal["fpr", "fdr"]


def contrast_zscore(
    image: np.ndarray,
    sim: SimData,
    contrast_name: str,
    **kwargs: None,
) -> tuple[np.ndarray, dict]:
    """Compute z-score of contrast_name.

    For now only a single contrast is supported.

    Parameters
    ----------
    image : np.ndarray
        4D image data.
    sim : SimData
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
    logger.debug(f"image_={image_.shape}, design matrix={design_matrix.shape}")
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
    height_control: HeightControl = "fpr",
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
    if len(alphas) == 1:
        return thresh_dict[alphas[0]]
    return thresh_dict


def get_scores(
    contrast: np.ndarray,
    ground_truth: np.ndarray,
    alphas: list[float],
    height_control: HeightControl = "fpr",
) -> dict[str, float]:
    """Get sklearn metrics scores.

    Parameters
    ----------
    thresh_map : np.ndarray
        Thresholded map.
    ground_truth : np.ndarray
        Ground truth map.
    alphas: list of float

    Returns
    -------
    scores : dict
        Dictionary of scores (accuracy, precision, recall, f1, jaccard)

    """
    gt_f = ground_truth.flatten()
    stats = {}
    stats = {"accuracy": [], "precision": [], "recall": [], "f1": [], "jaccard": []}
    for alpha in alphas:
        thresh_mapf = get_thresh_map(
            contrast, alpha=alpha, height_control=height_control
        ).flatten()
        stats["accuracy"].append(accuracy_score(gt_f, thresh_mapf))
        stats["precision"].append(precision_score(gt_f, thresh_mapf))
        stats["recall"].append(recall_score(gt_f, thresh_mapf))
        stats["f1"].append(f1_score(gt_f, thresh_mapf))
        stats["jaccard"].append(jaccard_score(gt_f, thresh_mapf))
    stats["alphas"] = list(alphas)
    stats["roc_auc"] = roc_auc_score(gt_f, contrast.flatten())
    fpr, tpr, thresholds = roc_curve(gt_f, contrast.flatten())
    stats["roc_fpr"] = list(fpr)
    stats["roc_tpr"] = list(tpr)
    stats["roc_thresh"] = list(thresholds)
    return stats
