"""Phantom Generation Handlers."""
from importlib.resources import files

import numpy as np

from simfmri.simulator.simulation import SimulationData
from simfmri.utils import validate_rng
from simfmri.utils.typing import RngType

from ..base import AbstractHandler
from ._big import generate_phantom, raster_phantom
from ._shepp_logan import mr_shepp_logan


class SheppLoganGeneratorHandler(AbstractHandler):
    """Create handler to generate the base phantom.

    Phantom generation should be the first step of the simulation.
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

    def __init__(
        self,
        B0: int | float = 7,
        roi_index: int = 10,
        dtype: np.dtype | str = np.float32,
    ):
        super().__init__()
        self.B0 = B0
        self.roi_index = roi_index
        self.dtype = dtype

    def _handle(self, sim: SimulationData) -> SimulationData:
        if len(sim.shape) != 3:
            raise ValueError("simulation shape should be 3D.")

        M0, T1, T2, labels = mr_shepp_logan(sim.shape, B0=self.B0, T2star=True)

        T2 = T2.astype(self.dtype)
        sim.data_ref = np.repeat(T2[None, ...], sim.n_frames, axis=0)
        sim.static_vol = T2.copy()
        sim.roi = labels == self.roi_index

        return sim


class BigPhantomGeneratorHandler(AbstractHandler):
    """Create Handler to create phantom based on bezier curves.

    Parameters
    ----------
    raster_osf
        rasterisation oversampling factor
    phantom_data
        location of the phantom parameter.
    """

    def __init__(
        self,
        raster_osf: int = 4,
        roi_index: int = 10,
        dtype: np.dtype | str = np.float32,
        phantom_data: str = "big",
    ):
        super().__init__()
        self.raster_osf = raster_osf
        self.phantom_data = phantom_data
        self.roi_index = roi_index
        self.dtype = dtype

    def _handle(self, sim: SimulationData) -> SimulationData:
        if len(sim.shape) > 2:
            raise ValueError("simulation shape should be 2D.")
        sim.static_vol = generate_phantom(
            sim.shape,
            raster_osf=self.raster_osf,
            phantom_data=self.phantom_data,
        )
        sim.data_ref = np.repeat(sim.static_vol[None, ...], sim.n_frames, axis=0)
        if self.roi_index is not None:
            sim.roi = (
                raster_phantom(sim.shape, self.phantom_data, weighting="label")
                == self.roi_index
            )
        return sim


class RoiDefinerHandler(AbstractHandler):
    """Define a Region of interest based on a bézier parametrization.

    Parameters
    ----------
    roi_data
        roi definition
    """

    def __init__(self, roi_data: list[dict] | dict = None, rng: RngType = None):
        super().__init__()
        if roi_data is None:
            roi_data = files("simfmri.simulator.handlers.phantom").joinpath(
                "big_phantom_roi.json"
            )
        self.roi_data = roi_data

    def _handle(self, sim: SimulationData) -> SimulationData:
        sim.roi = raster_phantom(
            sim.shape,
            phantom_data=self.roi_data,
            weighting="label",
        )
        sim.roi = sim.roi > 0
        return sim


class TextureAdderHandler(AbstractHandler):
    """Add texture to the image by playing a white noise.

    Parameters
    ----------
    var_texture
        relative factor to compute the noise variance.
    """

    def __init__(self, var_texture: float = 0.001, rng: RngType = None):
        super().__init__()
        self._var_texture = var_texture

    def _handle(self, sim: SimulationData) -> SimulationData:
        sigma_noise = self._var_texture * sim.data_ref[0]
        rng = validate_rng(sim.rng)
        sim.data_ref += sigma_noise * rng.standard_normal(
            sim.data_ref.shape[1:], dtype=sim.data_ref.dtype
        )

        return sim


class SlicerHandler(AbstractHandler):
    """Create an handler to get a 2D+T slice from a 3D+T simulation.

    Parameters
    ----------
    axis
        axis position to cut the slice
    index:
        index position on axis where the slice is perform.
    """

    def __init__(self, axis: int, index: int):
        super().__init__()
        if not (0 <= axis <= 2):
            raise ValueError("only 3D array are supported.")

        self.axis = axis
        self.index = index

    def _run_callback(self, old_sim: SimulationData, new_sim: SimulationData) -> None:
        """Notify that we are now in  2D."""
        self.log.warning("Simulation is now 2D")

    @property
    def slicer(self) -> tuple:
        """Return slicer operator."""
        base_slicer = [slice(None, None, None)] * 4
        base_slicer[self.axis + 1] = self.index
        return tuple(base_slicer)

    def _handle(self, sim: SimulationData) -> SimulationData:
        """Perform the slicing on all relevant data and update data_shape."""
        for data_type in ["data_ref", "_data_acq"]:
            if (array := getattr(sim, data_type)) is not None:
                setattr(sim, data_type, array[self.slicer])
        sim.roi = sim.roi[self.slicer[1:]]  # roi does not have frame dimension.
        new_shape = sim._meta.shape
        sim._meta.shape = tuple(s for ax, s in enumerate(new_shape) if ax != self.axis)
        return sim
