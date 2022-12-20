import numpy as np
from nilearn.glm.first_level import compute_regressor

from simfmri.simulator.simulation import Simulation

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
    event_condition : array-like of shape (3, n_events)
        yields description of events for this condition as a
        (onsets, durations, amplitudes) triplet
    hrf_model: str, default is 'glover'
        Choice for the HRF, FIR is not supported yet.
    oversampling : int, optional
        Oversampling factor to perform the convolution. Default=50.
    min_onset : float, optional
        Minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered.
        Default=-24.

    See also
    --------
    nilearn.compute_regressors
    """

    def __init__(
        self,
        event_condition: np.ndarray,
        roi: np.ndarray,
        bold_strength=0.01,
        hrf_model="glover",
        oversampling=50,
        min_onset=-24,
    ):

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
    def from_multi_event(cls, events, rois, **kwargs):

        if events.ndim != 3 or events.shape[1] != 3:
            raise ValueError("Event array should be of shape N_cond, 3, N_events")
        h_old = cls(events[0], rois[0], **kwargs)
        for event, roi_event in zip(events[1:], rois[1:]):
            h = cls(event, roi_event, **kwargs)
            h_old.set_next(h)
            h_old = h
        return h

    def _handle(self, sim: Simulation):

        if self._roi is None and sim.roi is None:
            raise ValueError("roi is not defined.")
        if sim.roi is None:
            sim.roi = self._roi.copy()
            roi = self._roi
        else:
            roi = sim.roi

        frame_times = sim.TR * np.ones(sim.n_frames)
        regressors, _ = compute_regressor(
            self._event_condition,
            self._hrf_model,
            frame_times,
            oversampling=self._oversampling,
            min_onset=self._min_onset,
        )
        regressors = np.squeeze(regressors)
        regressors /= np.max(regressors)

        sim.data_ref = roi * (1 + self._bold_strength) * regressors


def block_design(block_on, block_off, duration, offset=0):
    """Create a simple block design paradigm."""

    block_size = block_on + block_off

    event = []
    t = offset
    while t < duration:
        event.append((t, block_on, 1))
        t += block_size
    return np.array(event).T
