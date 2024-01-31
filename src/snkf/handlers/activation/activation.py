"""Handler to add activations."""

from __future__ import annotations

from typing import Literal, Mapping, get_args, Any

import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor  # type: ignore

from snkf.simulation.simulation import SimData, LazySimArray

from ..base import AbstractHandler, HandlerChain, requires_field
from ._block import block_design


HrfType = Literal["glover", "spm", None]


@requires_field("data_ref")
class ActivationHandler(AbstractHandler):
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

    See Also
    --------
    nilearn.compute_regressors
    """

    def __init__(
        self,
        event_condition: pd.DataFrame | np.ndarray,
        roi: np.ndarray | None,
        bold_strength: float = 0.02,
        hrf_model: HrfType = "glover",
        oversampling: int = 50,
        min_onset: float = -24.0,
    ):
        super().__init__()
        if hrf_model not in get_args(HrfType):
            raise ValueError(
                f"Unsupported HRF `{hrf_model}`, available are: {get_args(HrfType)}"
            )
        self._event_condition = event_condition
        self._hrf_model = hrf_model
        self._oversampling = oversampling
        self._bold_strength = bold_strength
        self._roi: np.ndarray | None = roi
        self._min_onset = min_onset

    @classmethod
    def from_multi_event(
        cls,
        events: np.ndarray,
        rois: Mapping[str, np.ndarray],
        bold_strength: float = 0.02,
        hrf_model: HrfType = "glover",
        oversampling: int = 50,
        min_onset: float = -24.0,
    ) -> HandlerChain:
        """
        Create a sequence of handler from a sequence of event and associated rois.

        The handlers are chained after prev_handler, and the last element of the chain
        is returned.

        Parameters
        ----------
        events
            Dataframe following the design_matrix structure
        rois
            dict (n_types_of_event, **)
        prev_handler
            The previous handler in the chain
        hrf_model
            str, default is 'glover'
            Choice for the HRF, FIR is not supported yet.
        oversampling : int, optional
            Oversampling factor to perform the convolution. Default=50.
        min_onset : float, optional
            Minimal onset relative to frame_times[0] (in seconds)
            events that start before frame_times[0] + min_onset are not considered.
            Default=-24.


        See Also
        --------
        nilearn.glm.first_level.make_first_level_design_matrix
            Design Matrix API.
        """
        if events.ndim != 3 or events.shape[1] != 3:
            raise ValueError("Event array should be of shape N_cond, 3, N_events")
        if len(events) != len(rois):
            raise ValueError("Event and rois should have the same first dimension.")
        h = HandlerChain()
        for event, roi_event in zip(events, rois, strict=True):
            h_new = cls(
                event,
                rois[roi_event],
                bold_strength,
                hrf_model,
                oversampling,
                min_onset,
            )
            h = h >> h_new
        return h

    def _handle(self, sim: SimData) -> SimData:
        if sim.roi is None and self._roi is not None:
            sim.roi = self._roi.copy()
            roi = self._roi
        elif sim.roi is not None:
            roi = sim.roi
        elif self._roi is None and sim.roi is None:
            raise ValueError("roi is not defined.")

        if np.sum(abs(roi)) == 0:
            raise ValueError("roi is empty.")
        regressor, _ = compute_regressor(
            self._event_condition[["onset", "duration", "modulation"]].to_numpy().T,
            self._hrf_model,
            np.linspace(0, sim.sim_time, sim.n_frames),
            oversampling=self._oversampling,
            min_onset=self._min_onset,
        )
        regressor = np.squeeze(regressor)
        regressor = regressor * self._bold_strength / regressor.max()
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
            df = df.concat(self._event_condition)
        else:
            df = self._event_condition
        sim._meta.extra_infos["events"] = df.to_dict()

        self.log.info(f"Simulated block activations at sim_tr={sim.sim_tr}s")
        return sim


class ActivationBlockHandler(ActivationHandler):
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
    snkf.utils.activations.block_design
        The helper function to create the block desing.
    """

    name = "activation-block"

    def __init__(
        cls,
        block_on: float,
        block_off: float,
        duration: float,
        offset: float = 0,
        event_name: str = "block_on",
        bold_strength: float = 0.02,
        hrf_model: HrfType = "glover",
        oversampling: int = 50,
        min_onset: float = -24.0,
    ):
        super().__init__(
            block_design(block_on, block_off, duration, offset, event_name),
            roi=None,
            bold_strength=bold_strength,
            hrf_model=hrf_model,
            oversampling=oversampling,
            min_onset=min_onset,
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
