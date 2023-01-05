"""Analysis module."""
import numpy as np

from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

from simfmri.simulator import SimulationData


def compute_t_test(
    sim: SimulationData,
    data_test: np.ndarray,
    alpha: float = 0.05,
    correction: str = None,
) -> np.ndarray:
    """
    Compute a T-Test on data_test based on the event of sim.extra_infos.

    Parameters
    ----------
    sim
        Simulation object
    data_test
        estimation of the data reconstructed from the simulation.
    alpha
        Threshold for the test
    correction
        Statistical correction to use (e.g. fdr)

    Returns
    -------
    numpy.ndarray
        a map of voxel detected as activating.

    See Also
    --------
    nilearn.glm.first_level.FirstLevelModel
        Backend for the glm computation.
    """
    # TODO compute the z-score using sim.extra_infos["events"]

    design_matrix = make_first_level_design_matrix()

    first_level_model = FirstLevelModel(t_r=sim.TR, hrf_model="glover")

    first_level_model.fit(data_test, design_matrices=design_matrix)
    first_level_model.compute_contrast("event", output_type="z-score")


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
