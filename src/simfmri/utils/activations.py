"""Utilities function for the activation."""
import numpy as np


def block_design(block_on: float, block_off: float, duration: float, offset: float = 0):
    """
    Create a simple block design paradigm.

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
    return np.array(event).T
