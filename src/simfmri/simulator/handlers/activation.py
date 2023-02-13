"""Handler to add activations."""
from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor
from simfmri.utils import block_design

from ..simulation import SimulationData
from .base import AbstractHandler

NILEARN_HRF = [
    "spm",
    # "spm + derivative",
    # "spm + derivative + dispersion",
    #    "fir",
    "glover",
    # "glover + derivative",
    # "glover + derivative + dispersion",
    None,
]


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
        event_condition: np.ndarray,
        roi: np.ndarray,
        bold_strength: float = 0.02,
        hrf_model: Literal[NILEARN_HRF] = "glover",
        oversampling: int = 50,
        min_onset: float = -24.0,
    ):
        super().__init__()
        if hrf_model not in NILEARN_HRF:
            raise ValueError(
                f"Unsupported HRF `{hrf_model}`, available are: {NILEARN_HRF}"
            )
        self._event_condition = event_condition
        self._hrf_model = hrf_model
        self._oversampling = oversampling
        self._bold_strength = bold_strength
        self._roi = roi
        self._min_onset = min_onset

    @classmethod
    def from_multi_event(
        cls,
        events: np.ndarray,
        rois: Mapping[str, np.ndarray],
        prev_handler: AbstractHandler,
        bold_strength: float = 0.02,
        hrf_model: Literal[NILEARN_HRF] = "glover",
        oversampling: int = 50,
        min_onset: float = -24.0,
    ) -> ActivationHandler:
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
        h_old = prev_handler
        for event, roi_event in zip(events, rois, strict=True):
            h = cls(
                event,
                roi_event,
                bold_strength,
                hrf_model,
                oversampling,
                min_onset,
            )
            h_old.set_next(h)
            h_old = h
        return h

    @classmethod
    def from_block_design(
        cls,
        block_on: float,
        block_off: float,
        duration: float,
        offset: float = 0,
        event_name: str = "block_on",
        **kwargs,
    ) -> ActivationHandler:
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
        simfmri.utils.activations.block_design
            The helper function to create the block desing.
        """
        return cls(
            block_design(block_on, block_off, duration, offset, event_name),
            roi=None,
            **kwargs,
        )

    def _handle(self, sim: SimulationData) -> SimulationData:

        if self._roi is None and sim.roi is None:
            raise ValueError("roi is not defined.")
        if sim.roi is None:
            sim.roi = self._roi.copy()
            roi = self._roi
        else:
            roi = sim.roi

        if np.sum(abs(roi)) == 0:
            raise ValueError("roi is empty.")
        frame_times = sim.TR * np.arange(sim.n_frames)
        regressor, _ = compute_regressor(
            self._event_condition[["onset", "duration", "modulation"]].to_numpy().T,
            self._hrf_model,
            frame_times,
            oversampling=self._oversampling,
            min_onset=self._min_onset,
        )
        regressor = np.squeeze(regressor)
        regressor = 1 + (regressor * self._bold_strength / regressor.max())
        # apply the activations
        sim.data_ref[:, roi] = sim.data_ref[:, roi] * regressor[:, np.newaxis]
        # update the experimental paradigm
        #
        if sim.extra_infos is None:
            sim._meta.extra_infos = {"events": self._event_condition, 'regressor': regressor}
            #changed here
        else:
            if isinstance(sim.extra_infos["events"], pd.DataFrame):
                sim._meta.extra_infos["events"].concat(self._event_condition)
        return sim
