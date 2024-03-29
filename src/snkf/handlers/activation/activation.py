"""Handler to add activations."""

from __future__ import annotations
from typing import Any
from enum import Enum
import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor  # type: ignore

from snkf.simulation.simulation import SimData, LazySimArray

from ..base import AbstractHandler, requires_field
from ._block import block_design


class HRF(str, Enum):
    """Available HRF models."""

    GLOVER = "glover"
    SPM = "spm"


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

    event_condition: pd.DataFrame | np.ndarray
    duration: float
    offset: float = 0
    event_name: str
    bold_strength: float
    hrf_model: HRF = HRF.GLOVER
    oversampling: int
    min_onset: float
    roi_threshold: float

    def _handle(self, sim: SimData) -> SimData:
        if sim.roi is None and self._roi is not None:
            sim.roi = self._roi.copy()
            roi = self._roi
        elif sim.roi is not None:
            roi = sim.roi
        elif self._roi is None and sim.roi is None:
            raise ValueError("roi is not defined.")

        if self.roi_threshold:  # optional binarization of the roi
            roi = roi > self.roi_threshold

        if np.sum(abs(roi)) == 0:
            raise ValueError("roi is empty.")
        # create HRF regressors.
        regressor, _ = compute_regressor(
            self.event_condition[["onset", "duration", "modulation"]].to_numpy().T,
            self.hrf_model,
            np.linspace(0, sim.sim_time, sim.n_frames),
            oversampling=self.oversampling,
            min_onset=self.min_onset,
        )
        regressor = np.squeeze(regressor)
        regressor = regressor * self.bold_strength / regressor.max()
        # apply the activations
        self.log.debug(f"Regressor values  {min(regressor)},{max(regressor)}")

        if isinstance(sim.data_ref, LazySimArray):
            sim.data_ref.apply(lazy_apply_regressor, regressor, roi)
        elif isinstance(sim.data_ref, np.ndarray):
            sim.data_ref[:, roi > 0] = sim.data_ref[:, roi > 0] * (
                (1 + regressor[:, np.newaxis]) * roi[roi > 0] + (1 - roi[roi > 0])
            )
        else:
            raise ValueError("sim.data_ref is not an array")

        if "events" in sim._meta.extra_infos.keys():
            df = pd.Dataframe(sim._meta.extra_infos["events"])
            df = df.concat(self.event_condition)
        else:
            df = self.event_condition
        sim._meta.extra_infos["events"] = df.to_dict()

        self.log.info(f"Simulated block activations at sim_tr={sim.sim_tr}s")
        return sim


@requires_field("data_ref")
class ActivationBlockHandler(ActivationMixin, AbstractHandler):
    """Create a activation handler from a block design.

    Parameters
    ----------
    block_on
        in seconds, the amount of time the stimuli is on
    block_off
        in seconds, the amount of time the stimuli is off (rest) after the on state.
    duration
        in seconds, the total amount of the experiments.
    offset
        in seconds, the starting point of the experiment.
    event_name
        name of the block event, default="block_on"

    See Also
    --------
    snkf.base.activations.block_design
        The helper function to create the block desing.
    """

    __handler_name__ = "activation-block"

    block_on: float
    block_off: float
    duration: float
    offset: float = 0
    event_name: str = "block_on"
    bold_strength: float = 0.02
    hrf_model: HRF = HRF.GLOVER
    oversampling: int = 50
    min_onset: float = -24.0
    roi_threshold: float = 0.0
    # event_condition: pd.DataFrame | np.ndarray = dataclasses.field(
    #    init=False, repr=False
    # )

    def __post_init__(self):
        super().__post_init__()
        self.event_condition = block_design(
            self.block_on,
            self.block_off,
            self.duration,
            self.offset,
            self.event_name,
        )


def lazy_apply_regressor(
    data: np.ndarray, regressor: np.ndarray, roi: np.ndarray, frame_idx: Any = None
) -> np.ndarray:
    """
    Lazy apply the regressor to the data.

    Parameters
    ----------
    data: np.ndarray
        frame data to apply the regressor.
    regressor: np.ndarray
        regressor data.
    roi: np.ndarray
        roi data.
    frame_idx: int
        frame index to apply the regressor.
    """
    data = np.copy(data)
    data[roi > 0] = data[roi > 0] * (
        (1 + regressor[frame_idx]) * roi[roi > 0] + (1 - roi[roi > 0])
    )
    return data
