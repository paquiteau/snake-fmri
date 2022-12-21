"""Acquisition Handler to genereate Kspace data from simulation."""
from typing import Callable, Union

import numpy as np
from fmri.operator.fourier import CartesianSpaceFourier
from simfmri.utils import get_smaps
from simfmri.utils.cartesian_sampling import get_cartesian_mask

from ..simulation import Simulation
from .base import AbstractHandler


class AcquistionHandler(AbstractHandler):
    """
    Simulate the acquisition of the data.

    Parameters
    ----------
    sampling_mask: ndarray or callable
    gen_smaps: boolean

    """

    def __init__(
        self, sampling_mask: Union(np.ndarray, Callable), gen_smaps: bool = True
    ):
        self._sampling_mask = sampling_mask
        self._gen_smaps = gen_smaps

        pass

    def _handle(self, sim: Simulation):
        if self._gen_smaps:
            sim.smaps = get_smaps(sim.shape, sim.n_coils)

        if callable(self._sampling_mask):
            mask = self._sampling_mask(sim)
        else:
            mask = self._sampling_mask
        fourier_op = CartesianSpaceFourier(
            sim.shape,
            mask=mask,
            n_coils=sim.n_coils,
            n_frames=sim.n_frames,
            smaps=sim.smaps,
            n_jobs=-1,
        )

        sim.kspace_data = fourier_op.op(sim.data_acq)
        sim.kspace_mask = mask
        pass

    @classmethod
    def vds(
        cls,
        acs: Union(float, int),
        accel: int,
        constant: bool = True,
        gen_smaps: bool = True,
    ):
        """
        Generate Acquisition with Variable density sampling (vds).

        Parameters
        ----------
        acs: autocalibration line number (int) or proportion (float)
        accel: Acceleration factor
        constant: If true, the sampling pattern is repeated across time
        gen_smaps: If true, smaps are generated, and used in the acquisition.
        """

        def sampling_mask(sim: Simulation):
            return get_cartesian_mask(
                shape=sim.shape,
                n_frames=sim.n_frames,
                constant=constant,
                center_prop=acs,
                accel=accel,
                pdf="gaussian",
            )

        return cls(sampling_mask=sampling_mask, gen_smaps=gen_smaps)
