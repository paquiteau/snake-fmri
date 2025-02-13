"""Activation Handler."""

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ...._meta import LogMixin
from ...phantom import Phantom, DynamicData
from ...simulation import SimConfig
from ..base import AbstractHandler
from ..utils import apply_weights
from .roi import BRAINWEB_OCCIPITAL_ROI, get_indices_inside_ellipsoid
from .bold import get_bold, block_design, get_event_ts


class ActivationMixin(LogMixin):
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
        If greater than 0, the roi becomes a binary mask, with roi_threshold
        as separation.

    See Also
    --------
    nilearn.compute_regressors
    """

    event_condition: pd.DataFrame | np.ndarray
    duration: float
    offset: float = 0
    event_name: str
    roi_tissue_name: str = "ROI"
    delta_r2s: float = 1000.0
    hrf_model: str = "glover"
    oversampling: int = 10
    min_onset: float = -24.0
    roi_threshold: float = 0.0
    base_tissue_name = "gm"

    def get_static(self, phantom: Phantom, sim_config: SimConfig) -> Phantom:
        """Get the static ROI."""
        tissue_index = phantom.labels == self.base_tissue_name
        # shape: tuple[int, ...] | None = phantom.tissues_mask.shape[1:]
        if tissue_index.sum() == 0:
            raise ValueError(
                f"Tissue {self.base_tissue_name} not found in the phantom."
            )
        roi = phantom.masks[tissue_index].squeeze()
        occ_roi = BRAINWEB_OCCIPITAL_ROI.copy()
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
        new_phantom = phantom.add_tissue(
            self.roi_tissue_name,
            roi,
            phantom.props[tissue_index, :],
            phantom.name + "-roi",
        )

        return new_phantom

    def get_dynamic(self, phantom: Phantom, sim_conf: SimConfig) -> DynamicData:
        """Get dynamic time series for adding Activations."""
        bold_strength = sim_conf.seq.TE / self.delta_r2s

        self.log.info("Computed BOLD Strength: %s", bold_strength)
        bold = get_bold(
            sim_conf.sim_tr_ms,
            sim_conf.max_sim_time,
            self.event_condition,
            self.hrf_model,
            self.oversampling,
            self.min_onset,
            bold_strength,
        ).squeeze()
        events = get_event_ts(
            self.event_condition,
            sim_conf.max_sim_time,
            sim_conf.sim_tr_ms,
            self.min_onset,
        ).squeeze()
        return DynamicData(
            name="-".join(["activation", self.event_name]),
            data=np.concatenate([bold[None, :], events[None, :]]),
            func=self.apply_weights,
        )

    @staticmethod
    def apply_weights(phantom: Phantom, data: NDArray, time_idx: int) -> Phantom:
        """Apply weights to the ROI."""
        return apply_weights(phantom, "ROI", data[0], time_idx)


class BlockActivationHandler(ActivationMixin, AbstractHandler):
    """Activation Handler with block design.

    Parameters
    ----------
    block_on: float
        time the block activation is on.
    block_off: float
        time the block activation is off
    duration: float
        Total duration of the pattern in seconds
    offset: float, default 0
        Start time of the pattern in seconds
    roi_tissue_name: str, default "ROI"
        Name of the ROI tissue
    event_name: str, default "block_on"
        Name of the event
    delta_r2s: float, default 1000.0
        Delta R2s value
    hrf_model: str, default "glover"
        HRF model
    oversampling: int, default 50
        Oversampling factor
    min_onset: float, default -24.0
        Minimal onset
    roi_threshold: float, default 0.0
        ROI threshold

    Notes
    -----
    See Also the GLM module of Nilearn.


    """

    __handler_name__ = "activation-block"

    block_on: float
    block_off: float
    duration: float
    offset: float = 0
    roi_tissue_name: str = "ROI"
    event_name: str = "block_on"
    delta_r2s: float = 1000.0
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
