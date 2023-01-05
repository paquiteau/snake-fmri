"""Analysis module."""
import numpy as np
from typing import Literal

from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm import threshold_stats_img

from simfmri.simulator import SimulationData


def compute_test(
    sim: SimulationData,
    data_test: np.ndarray,
    contrast_name: str,
    stat_type: Literal["t", "F"] = "t",
    alpha: float = 0.05,
    height_control: str = "fpr",
) -> np.ndarray:
    """
    Compute a T-Test on data_test based on the event of sim.extra_infos.

    Parameters
    ----------
    sim
        Simulation object
    data_test
        estimation of the data reconstructed from the simulation.
    contrast_name
        name or list of name of the contrast to test for.
    stat_type
        "t" for t-test, "F" for F-test, default="t",
    alpha
        Threshold for the test
    height_control
        Statistical correction to use (e.g. fpr or fdr)

    Returns
    -------
    numpy.ndarray
        a map of voxel detected as activating.

    See Also
    --------
    nilearn.glm.first_level.FirstLevelModel
        Backend for the glm computation.
    """
    # instantiate model
    design_matrix = make_first_level_design_matrix(
        frames_times=np.arange(sim.n_frames) * sim.TR, events=sim.extra_infos["events"]
    )
    first_level_model = FirstLevelModel(t_r=sim.TR, hrf_model="glover")

    # fit the model with all confounds
    first_level_model.fit(data_test, design_matrices=design_matrix)

    # extract classification.
    contrast = first_level_model.compute_contrast(
        contrast_name, stat_type=stat_type, output_type="z-score"
    )
    thresh = threshold_stats_img(contrast, alpha=alpha, height_control=height_control)

    return contrast > thresh


def compute_stats(estimation: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Compute confusion statistics.

    Parameters
    ----------
    estimation
        estimation of the classification
    ground_truth
        ground truth map for the classification
    """
    neg = np.sum(ground_truth == 0)
    pos = np.sum(ground_truth > 0)

    f_neg = np.sum(estimation == 0 & ground_truth > 0)
    t_neg = np.sum(estimation == 0 & ground_truth == 0)
    f_pos = np.sum(estimation == 1 & ground_truth == 0)
    t_pos = np.sum(estimation == 1 & ground_truth > 0)

    stats = dict()
    stats["TPR"] = t_pos / pos  # sensitivity
    stats["TNR"] = t_neg / neg  # specificity
    stats["PPV"] = t_pos / (f_pos + t_pos)  # precision
    stats["FPR"] = f_pos / (f_pos + t_neg)  # false positive rate
    stats["FNR"] = f_neg / (t_pos + f_neg)  # false negative rate
    stats["FDR"] = f_pos / (f_pos + t_pos)  # false discovery rate
    return stats
