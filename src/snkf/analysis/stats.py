"""Analysis module."""

import logging
from typing import Literal
from numpy.typing import NDArray
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np

from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm import compute_contrast, expression_to_contrast_vector
from nilearn.glm.thresholding import fdr_threshold
from scipy.stats import norm

from snkf.simulation import SimData

logger = logging.getLogger(__name__)

HeightControl = Literal["fpr", "fdr"]


def contrast_zscore(
    image: np.ndarray,
    sim: SimData,
    contrast_name: str,
    **kwargs: None,
) -> np.ndarray:
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
        events=pd.DataFrame(sim.extra_infos["events"]),
        drift_model=sim.extra_infos.get("drift_model", None),
    )
    # Create a mask from reference data (not ideal, but best)
    mask = sim.static_vol > 0
    image_ = abs(image)[..., mask].squeeze()
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
    stats = {}
    gt_f = ground_truth.flatten() >= 0.5
    P = np.sum(gt_f)
    N = gt_f.size - P

    fpr, tpr, thresholds = roc_curve(gt_f, contrast.flatten())
    stats["fpr"] = fpr.tolist()
    stats["tpr"] = tpr.tolist()
    stats["tresh"] = thresholds.tolist()
    stats["tp"] = np.int_(tpr * P).tolist()
    stats["fp"] = np.int_(fpr * N).tolist()
    stats["tn"] = np.int_(N * (1 - fpr)).tolist()
    stats["fn"] = np.int_(P * (1 - tpr)).tolist()
    return stats


def bacc(tpr: NDArray, fpr: NDArray, adjusted: bool = False) -> NDArray:
    """Compute Balanced Accuracy from TPR and FPR."""
    bacc = (tpr + 1 - fpr) / 2
    if adjusted:
        return (bacc - 0.5) * 2
    return bacc


def mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    """Compute Matthew's Correlation Coefficient."""
    tpr = tp / (tp + fn)
    tnr = tn / (fp + tn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    fnr = 1 - tpr
    fpr = 1 - tnr
    for_ = 1 - npv
    fdr = 1 - ppv

    return np.sqrt(tpr * tnr * ppv * npv) - np.sqrt(fnr * fpr * for_ * fdr)
