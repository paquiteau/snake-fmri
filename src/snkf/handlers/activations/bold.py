"""BOLD Signal utilities."""

import pandas as pd
import numpy as np
from nilearn.glm.first_level.hemodynamic_models import (
    _hrf_kernel,
    _sample_condition,
    _resample_regressor,
)


def block_design(
    block_on: float,
    block_off: float,
    duration: float,
    offset: float = 0,
    event_name: str = "block_on",
) -> pd.DataFrame:
    """
    Create a simple block design paradigm.

    Parameters
    ----------
    block_on: float
        in seconds, the amount of time the stimuli is on
    block_off: float
        in seconds, the amount of time the stimuli is off (rest) after the on state.
    duration: float
        in seconds, the total amount of the experiments.
    offset: float
        in seconds, the starting point of the experiment, default=0.
    event_name: str
        name of the block event, default="block_on"

    Returns
    -------
    pd.DataFrame
        the data frame corresponding to a block design.

    Notes
    -----
    The design is as follows ::

                |---------|          |----------|       |------>
       |--------|         |----------|          |-------|
         offset   block_on  block_off  block_on   ...

    And repeats until `duration` is reached.
    """
    block_size = block_on + block_off

    event = []
    t = offset
    while t < duration:
        event.append((t, block_on, 1))
        t += block_size
    events = np.array(event)

    return pd.DataFrame(
        {
            "trial_type": event_name,
            "onset": events[:, 0],
            "duration": events[:, 1],
            "amplitude": events[:, 2],
        }
    )


def get_bold(
    tr_s: float,
    max_time: float,
    event_condition: np.ndarray | pd.DataFrame,
    hrf_model: str,
    oversampling: int,
    min_onset: float,
    bold_strength: float,
) -> np.ndarray:
    """Convolve the HRF with the event condition to generate the BOLD signal.

    Parameters
    ----------
    frame_times:
        array-like of shape (n_times,)
        The timing of the acquisition
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
    duration:
        Duration of the event in seconds.
    offset:
        Offset of the event in seconds.
    bold_strength:
        Strength of the BOLD signal.

    Returns
    -------
    np.ndarray
        The convolved HRF with the event condition.
    """
    frame_times = np.arange(0, max_time * 1000, tr_s)
    hr_regressor, frame_times_hr = _sample_condition(
        [
            event_condition["onset"].values,
            event_condition["duration"].values,
            event_condition["amplitude"].values,
        ],
        frame_times,
        oversampling=oversampling,
        min_onset=min_onset,
    )

    # 2. create the  hrf model(s)
    hkernel = _hrf_kernel(hrf_model, tr_s, oversampling)

    # 3. convolve the regressor and hrf, and downsample the regressor
    conv_reg = np.array(
        [np.convolve(hr_regressor, h)[: hr_regressor.size] for h in hkernel]
    )
    computed_regressors = _resample_regressor(conv_reg, frame_times_hr, frame_times)
    return computed_regressors
