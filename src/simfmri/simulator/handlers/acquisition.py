from .base import AbstractHandler
from fmri.operator.fourier import CartesianSpaceFourier
from typing import Union
import numpy as np
from simfmri.utils import get_smaps


class AcquistionHandler(AbstractHandler):
    """Simulate the acquisition of the data.

    Parameters
    ----------

    """

    def __init__(self, sampling_mask: np.ndarray, gen_smaps=True):
        self._sampling_mask = sampling_mask
        self._gen_smaps = gen_smaps

        pass

    def _handle(self, sim):
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
    def const_vds(cls, acs: Union(float, int)):

        pass

    @classmethod
    def variable_vds(
        cls,
        acs: Union(float, int),
    ):
        pass

    @classmethod
    def const_us(cls, acs: int, R: int = 2):
        pass
