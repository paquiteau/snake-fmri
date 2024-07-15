"""Activation Handler."""

from collections.abc import Mapping, Callable
import numpy as np
import pandas as pd

from ...phantom import Phantom, DynamicData
from ...simulation import SimConfig
from ..base import AbstractHandler
from ..utils import apply_weights
from .roi import BRAINWEB_OCCIPITAL_ROI, get_indices_inside_ellipsoid
from .bold import get_bold, block_design


class ActivationMixin:
    """Add activation inside the region of interest. for a single type of event.

    Parameters
    ----------
    event_condition:
        array-like of shape (3, n_events)
        yields description of events for this condition as a
        (onsets, durations, amplitudes) triplet
    hrf_model:
        Choice for the HRF, FIR is not
    oversampling:
        Oversampling factor to perform the convolution. Default=50.
    min_onset:
        Minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered.
        Default=-24.
    roi_threshold: float, default 0.0
        If greated than 0, the roi becomes a binary mask, with roi_threshold
        as separation.

    See Also
    --------
    nilearn.compute_regressors
    """

    __handler_name__ = "activation"

    event_condition: pd.DataFrame | np.ndarray
    duration: float
    offset: float = 0
    event_name: str
    bold_strength: float = 0.02
    hrf_model: str = "glover"
    oversampling: int = 10
    min_onset: float = -24.0
    roi_threshold: float = 0.0
    base_tissue_name = "gm"

    def get_static(self, phantom: Phantom, sim_config: SimConfig) -> Phantom:
        """Get the static ROI."""
        tissue_index = phantom.tissue_label == self.base_tissue_name
        # shape: tuple[int, ...] | None = phantom.tissues_mask.shape[1:]
        if tissue_index.sum() == 0:
            raise ValueError(
                f"Tissue {self.base_tissue_name} not found in the phantom."
            )
        roi = phantom.tissue_masks[tissue_index].squeeze()
        occ_roi = BRAINWEB_OCCIPITAL_ROI.copy()
        # if self.bbox:
        #     self.log.debug("ROI shape was ", occ_roi)
        #     scaled_bbox = [0] * 6
        #     for i in range(3):
        #         scaled_bbox[2 * i] = (
        #             int(self.bbox[2 * i] * occ_roi["shape"][i])
        #             if self.bbox[2 * i]
        #             else 0
        #         )
        #         scaled_bbox[2 * i + 1] = (
        #             int(self.bbox[2 * i + 1] * occ_roi["shape"][i])
        #             if self.bbox[2 * i + 1]
        #             else occ_roi["shape"][i]
        #         )
        #         if scaled_bbox[2 * i + 1] < 0:
        #             scaled_bbox[2 * i + 1] += occ_roi["shape"][i]

        #     # replace None value by boundaries (0 or value)
        #     # and shift the box
        #     occ_roi["shape"] = (
        #         scaled_bbox[1] - scaled_bbox[0],
        #         scaled_bbox[3] - scaled_bbox[2],
        #         scaled_bbox[5] - scaled_bbox[4],
        #     )
        #     occ_roi["center"] = (
        #         occ_roi["center"][0] - scaled_bbox[0],
        #         occ_roi["center"][1] - scaled_bbox[2],
        #         occ_roi["center"][2] - scaled_bbox[4],
        #     )
        roi_zoom = np.array(roi.shape) / np.array(occ_roi["shape"])
        self.log.debug(
            "ROI parameters (orig, target, zoom) %s, %s, %s",
            occ_roi["shape"],
            roi.shape,
            roi_zoom,
        )
        ellipsoid = get_indices_inside_ellipsoid(
            roi.shape,
            center=np.array(occ_roi["center"]) * roi_zoom,
            semi_axes_lengths=np.array(occ_roi["semi_axes_lengths"]) * roi_zoom,
            euler_angles=occ_roi["euler_angles"],
        )
        roi[~ellipsoid] = 0

        # update the phantom
        new_phantom = Phantom(
            phantom.name + "-roi",
            tissue_masks=np.concatenate((phantom.tissue_masks, roi[None, ...]), axis=0),
            tissue_label=np.concatenate((phantom.tissue_label, ["ROI"])),
            tissue_properties=np.concatenate(
                (phantom.tissue_properties, phantom.tissue_properties[tissue_index, :]),
                axis=0,
            ),
        )
        return new_phantom

    def get_dynamic(
        self, phantom: Phantom, sim_conf: SimConfig
    ) -> (Mapping[str, np.ndarray], Callable):
        """Get dynamic time series for adding Activations."""
        return DynamicData(
            name="-".join(["activation", self.event_name]),
            data=get_bold(
                sim_conf.sim_tr_ms,
                sim_conf.max_sim_time,
                self.event_condition,
                self.hrf_model,
                self.oversampling,
                self.min_onset,
                self.bold_strength,
            ).T,
            func=self.apply_weights,
        )


class BlockActivationHandler(ActivationMixin, AbstractHandler):
    """Activation Handler with block design."""

    __handler_name__ = "activation-block"

    block_on: float
    block_off: float
    duration: float
    offset: float = 0
    event_name: str = "block_on"
    bold_strength: float = 0.02
    hrf_model: str = "glover"
    oversampling: int = 50
    min_onset: float = -24.0
    roi_threshold: float = 0.0

    def __post_init__(self):
        self.event_condition = block_design(
            self.block_on,
            self.block_off,
            self.duration,
            self.offset,
            self.event_name,
        )

    @staticmethod
    def apply_weights(phantom, data, time_idx):
        return apply_weights(phantom, "ROI", data, time_idx)
