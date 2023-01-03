from typing import Union
from .base import AbstractHandler
from ..simulation import SimulationData


from ...utils.phantom import mr_shepp_logan


class SheppLoganPhantomGeneratorHandler(AbstractHandler):
    """Handler to create the base phantom.

    phantom generation should be the first step of the simulation.
    Moreover, it only accept 3D shape.

    Parameters
    ----------
    B0:
        main magnetic field intensity.
    roi_index:
        the index for the region of interest.
        It is one of the ellipsis defined in the phantom.

    See Also
    --------
    utils.phantom: Module defining the shepp logan phantom.
    """

    def __init__(self, B0: Union[int, float] = 7, roi_index: int = 10):
        self.B0 = B0
        self.roi_index = roi_index

    def _handle(self, sim: SimulationData):
        if len(sim.shape) != 3:
            raise ValueError("simulation shape should be 3D.")

        M0, T1, T2, labels = mr_shepp_logan(sim.shape, B0=self.B0, T2star=True)

        sim.data_ref = T2
        sim.data = T2.copy()

        sim.roi = labels == self.roi_index

        return sim
