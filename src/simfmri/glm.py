"""Analysis module."""
import numpy as np

# from nilearn.glm.first_level import the good function
#


def make_t_test(sim, data_rec) -> np.ndarray:
    """Get the z-score  for the reconstructed data.

    Parameters
    ----------
    sim
        Simulation object
    data_rec
        estimation of the data reconstructed from the simulation.

    Returns
    -------
    a statistical map of z-score for the effect of interest.

    """


def compute_stats(stat_map, ground_truth, p_value):
    """Compute confusion statistics."""
    estimation = stat_map > p_value  # TODO Change this.
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
