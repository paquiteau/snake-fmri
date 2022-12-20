from .base import AbstractHandler
from fmri.operator.fourier import CartesianSpaceFourier
from typing import Union


class AcquistionHandler(AbstractHandler):
    """Simulate the acquisition of the data."""

    def _handle(self, sim):
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
