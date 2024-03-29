"""Utilities function for the activation."""

import numpy as np
import pandas as pd


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
    The design is as follows
    ::

       |--------|---------|----------|----------|-----|------>
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
            "modulation": events[:, 2],
        }
    )
