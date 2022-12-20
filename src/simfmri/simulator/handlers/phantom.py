from typing import Union
from .base import AbstractHandler
from ..simulation import Simulation


from ...utils.phantom import mr_shepp_logan


class SheppLoganPhantomGeneratorHandler(AbstractHandler):
    """Handler to create the base phantom.

    phantom generation should be the first step of the simulation.
    Moreover, it only accept 3D shape.

    Parameters
    ----------
    B0: main magnetic field intensity.

    """

    def __init__(self, B0: Union(int, float) = 7, roi_index=10):
        self.B0 = B0
        self.roi_index = roi_index

    def _handle(self, sim: Simulation):
        if len(sim.shape) != 3:
            raise ValueError("simulation shape should be 3D.")

        M0, T1, T2, labels = mr_shepp_logan(sim.shape, B0=self.B0, T2star=True)

        sim.data_ref = T2
        sim.roi = labels == self.roi_index
