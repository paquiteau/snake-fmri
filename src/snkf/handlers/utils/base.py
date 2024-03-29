# /usr/bin/env python3
"""Utility Handlers for the SNAKE-fMRI package."""

from snkf.handlers.base import AbstractHandler
from snkf.simulation import SimData


class KspaceFrameSlicerHandler(AbstractHandler):
    """Hanlder that cut a simulation to a specific frame idx.

    Temporaly slice/resample the simulation according to kspace-volume time resolution.
    If no acquisition has been performed, it is required to have the tr_vol parameters.
    This parameters can be obtained by using an acquisition handler with mock=True as
    an extra parameter.
    """

    __handler_name__ = "slicer-frame"

    start: int = 0
    stop: int = -1
    step: int = 1

    def _handle(self, sim: SimData) -> SimData:
        raise NotImplementedError
        # if kspace-data and kspace_mask are available
        ...
        # elif we have extra_infos field with data
        ...
        # else raise Error

        # Do the slicing
        # data_ref, data_acq
        # kspace_data, kspace_mask  if available.
        # update sim_time, n_frames etc ...
