"""Analysis module."""
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Literal

from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm import threshold_stats_img

from simfmri.simulator import SimulationData


def compute_test(
    sim: SimulationData,
    data_test: np.ndarray,
    contrast_name: str,
    stat_type: Literal["t", "F"] = "t",
    alpha: float | list[float] = 0.05,
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
        frame_times=np.arange(sim.n_frames) * sim.TR,
        events=sim.extra_infos["events"],
        drift_model=sim.extra_infos.get("drift_model", None),
    )
    first_level_model = FirstLevelModel(t_r=sim.TR, hrf_model="glover", mask_img=False)

    # fit the model with all confounds
    if isinstance(data_test, np.ndarray):
        data_test = nib.Nifti1Image(abs(data_test), affine=np.eye(4))
    first_level_model.fit(data_test, design_matrices=design_matrix)

    # extract classification.
    contrast = first_level_model.compute_contrast(
        contrast_name, stat_type=stat_type, output_type="z_score"
    )

    threshold_map, threshold = threshold_stats_img(
        contrast,
        alpha=alpha,
        height_control=height_control,
    )

    return threshold_map.get_fdata(), design_matrix, contrast.get_fdata()


def compute_confusion(estimation: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Compute confusion statistics.

    Parameters
    ----------
    estimation
        estimation of the classification
    ground_truth
        ground truth map for the classification
    """
    f_neg = np.sum((estimation == 0) & (ground_truth > 0))
    t_neg = np.sum((estimation == 0) & (ground_truth == 0))
    f_pos = np.sum((estimation > 0) & (ground_truth == 0))
    t_pos = np.sum((estimation > 0) & (ground_truth > 0))

    # casting to remove any numpy dtype sugar.
    confusion = {
        "f_neg": int(f_neg),
        "t_neg": int(t_neg),
        "f_pos": int(f_pos),
        "t_pos": int(t_pos),
    }
    return confusion


def compute_confusion_stats(
    f_neg: int, t_neg: int, f_pos: int, t_pos: int
) -> dict[str, float]:
    """Compute the confusion statistics."""
    stats = dict()
    stats["TPR"] = t_pos / ((t_pos + f_neg) or 1)  # sensitivity
    stats["TNR"] = t_neg / ((t_neg + f_pos) or 1)  # specificity
    stats["PPV"] = t_pos / ((f_pos + t_pos) or 1)  # precision
    stats["FPR"] = f_pos / ((f_pos + t_neg) or 1)  # false positive rate
    stats["FNR"] = f_neg / ((t_pos + f_neg) or 1)  # false negative rate
    stats["FDR"] = f_pos / ((f_pos + t_pos) or 1)  # false discovery rate

    return stats


def append_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the confusion stastistic over the row of a dataframe."""
    stats = []
    for _idx, row in df.iterrows():
        stats.append(
            compute_confusion_stats(
                row["f_neg"], row["t_neg"], row["f_pos"], row["t_pos"]
            )
        )

    return pd.concat([df, pd.DataFrame(stats)], axis=1)
