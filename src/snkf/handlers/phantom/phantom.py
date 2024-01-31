"""Phantom Generation Handlers."""
from importlib.resources import files
import os

import numpy as np
from numpy.typing import DTypeLike, NDArray
from joblib.hashing import hash as jbhash

from brainweb_dl import get_mri
from brainweb_dl._brainweb import get_brainweb_dir

from snkf.simulation import SimData, LazySimArray
from snkf.utils import validate_rng, real_type
from snkf.utils.typing import RngType

from ..base import AbstractHandler, requires_field
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

    name = "phantom-shepp_logan"

    def __init__(
        self,
        B0: int | float = 7,
        roi_index: int = 10,
        dtype: DTypeLike = np.float32,
    ):
        super().__init__()
        self.B0 = B0
        self.roi_index = roi_index
        self.dtype = dtype

    def _handle(self, sim: SimData) -> SimData:
        if len(sim.shape) != 3:
            raise ValueError("simulation shape should be 3D.")

        M0, T1, T2, labels = mr_shepp_logan(sim.shape, B0=self.B0, T2star=True)

        T2 = T2.astype(self.dtype)
        if sim.lazy:
            sim.data_ref = LazySimArray(T2, sim.n_frames)
        else:
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

    name = "phantom-big"

    def __init__(
        self,
        raster_osf: int = 4,
        roi_index: int = 10,
        dtype: DTypeLike = np.float32,
        phantom_data: str = "big",
    ):
        super().__init__()
        self.raster_osf = raster_osf
        self.phantom_data = phantom_data
        self.roi_index = roi_index
        self.dtype = dtype

    def _handle(self, sim: SimData) -> SimData:
        if len(sim.shape) > 2:
            raise ValueError("simulation shape should be 2D.")
        sim.static_vol = generate_phantom(
            sim.shape,
            raster_osf=self.raster_osf,
            phantom_data=self.phantom_data,
        )
        if sim.lazy:
            sim.data_ref = LazySimArray(sim.static_vol, sim.n_frames)
        else:
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

    name = "phantom-roi"
    roi_data: str | dict | list[dict] | os.PathLike

    def __init__(
        self,
        roi_data: str | dict | list[dict] | os.PathLike | None = None,
        rng: RngType = None,
    ):
        super().__init__()
        if roi_data is None:
            self.roi_data = str(files("snkf.handlers.phantom") / "big_phantom_roi.json")
        else:
            self.roi_data = roi_data

    def _handle(self, sim: SimData) -> SimData:
        sim.roi = raster_phantom(
            sim.shape,
            phantom_data=self.roi_data,
            weighting="label",
        )
        sim.roi = sim.roi > 0
        return sim


class BrainwebPhantomHandler(AbstractHandler):
    """Handler to load brainweb phantom.

    Parameters
    ----------
    subject_id
        subject id to load.
    brainweb_folder
        folder where the brainweb data are stored.
    roi
        region of interest to extract.
    """

    name = "phantom-brainweb"

    def __init__(
        self,
        subject_id: int,
        brainweb_folder: os.PathLike | None = None,
        roi: int = 1,
        bbox: tuple[int, int, int, int, int, int] | None = None,
        res: float | tuple[float, float, float] | None = None,
        force: bool = False,
        rng: RngType = None,
    ):
        super().__init__()
        self.sub_id = subject_id
        self.brainweb_folder = brainweb_folder
        self.roi = roi
        self.bbox = bbox
        self.force = force
        self.res = res
        self.rng = rng

    def _handle(self, sim: SimData) -> SimData:
        # TODO hash and cache config with all the parameters of get_mri
        # do this for both static_vol and roi.
        # save to brainweb_folder
        bw_dir = get_brainweb_dir(self.brainweb_folder)
        self.log.debug(
            f"hash config {self.sub_id, sim.shape, self.bbox, sim.rng, self.res, }"
        )
        static_hash = jbhash(
            str(("static", self.sub_id, sim.shape, self.bbox, sim.rng, self.res))
        )
        roi_hash = jbhash(str((self.sub_id, sim.shape, self.bbox, self.res)))

        static_path = bw_dir / (static_hash + ".npy")
        roi_path = bw_dir / (roi_hash + ".npy")

        static_vol: NDArray
        if os.path.exists(static_path) and not self.force:
            static_vol = np.load(static_path)
        else:
            self.log.warning(f"static volume was not cached at {static_path}.")
            static_vol = self._make_static_vol(sim)
            os.makedirs(static_path.parent, exist_ok=True)
            np.save(static_path, static_vol)

        roi: NDArray
        if os.path.exists(roi_path) and not self.force:
            roi = np.load(roi_path)
        else:
            roi = self._make_roi(sim)
            np.save(roi_path, roi)

        # update the simulation
        if -1 in sim.shape:
            old_shape = sim.shape
            sim._meta.shape = static_vol.shape
            self.log.warning(f"shape was implicit {old_shape}, it is now {sim.shape}.")
        if sim.fov is None or sim.fov == (-1, -1, -1):
            sim._meta.fov = (0.181, 0.217, 0.181)  # standard fov for brainweb.
        if self.bbox:
            # update FOV
            new_fov = [0.0] * 3
            for i in range(3):
                bmin = self.bbox[2 * i]
                bmax = self.bbox[2 * i + 1]
                bmin = bmin if bmin else 0
                bmax = bmax if bmax else 1
                bmax = 1 + bmax if bmax < 0 else bmax

                new_fov[i] = sim.fov[i] * (bmax - bmin)

            self.log.warning(f"sim.fov was  {sim.fov}, it is now {new_fov}.")
            sim._meta.fov = tuple(new_fov)

        sim.static_vol = static_vol
        sim.roi = roi

        # create the data ref field.
        if sim.lazy:
            sim.data_ref = LazySimArray(static_vol, sim.n_frames)
        else:
            sim.data_ref = np.repeat(static_vol[None, ...], sim.n_frames, axis=0)

        self.log.debug(f"roi shape: {sim.roi.shape}")
        self.log.debug(f"data_ref shape: {sim.data_ref.shape}")
        return sim

    def _make_static_vol(self, sim: SimData) -> np.ndarray:
        self.log.debug("Using brainweb_dl for data generation.")
        shape: tuple[int, ...] | None = sim.shape
        if shape == (-1, -1, -1):
            shape = None
        return get_mri(
            self.sub_id,
            brainweb_dir=self.brainweb_folder,
            contrast="T2*",
            rng=sim.rng or self.rng,
            shape=shape,
            output_res=self.res,
            bbox=self.bbox,
        )

    def _make_roi(self, sim: SimData) -> np.ndarray:
        from ._brainweb import (
            get_indices_inside_ellipsoid,
            BRAINWEB_OCCIPITAL_ROI,
        )

        shape: tuple[int, ...] | None = sim.shape
        if shape == (-1, -1, -1):
            shape = None
        roi = get_mri(
            self.sub_id,
            brainweb_dir=self.brainweb_folder,
            contrast="fuzzy",
            shape=shape,
            bbox=self.bbox,
            output_res=self.res,
        )[..., 2]

        occ_roi = BRAINWEB_OCCIPITAL_ROI.copy()
        if self.bbox:
            self.log.debug("ROI shape was ", occ_roi)
            scaled_bbox = [0] * 6
            for i in range(3):
                scaled_bbox[2 * i] = (
                    int(self.bbox[2 * i] * occ_roi["shape"][i])
                    if self.bbox[2 * i]
                    else 0
                )
                scaled_bbox[2 * i + 1] = (
                    int(self.bbox[2 * i + 1] * occ_roi["shape"][i])
                    if self.bbox[2 * i + 1]
                    else occ_roi["shape"][i]
                )
                if scaled_bbox[2 * i + 1] < 0:
                    scaled_bbox[2 * i + 1] += occ_roi["shape"][i]

            # replace None value by boundaries (0 or value)
            # and shift the box
            occ_roi["shape"] = (
                scaled_bbox[1] - scaled_bbox[0],
                scaled_bbox[3] - scaled_bbox[2],
                scaled_bbox[5] - scaled_bbox[4],
            )
            occ_roi["center"] = (
                occ_roi["center"][0] - scaled_bbox[0],
                occ_roi["center"][1] - scaled_bbox[2],
                occ_roi["center"][2] - scaled_bbox[4],
            )
        self.log.debug("ROI shape is ", occ_roi)
        roi_zoom = np.array(roi.shape) / np.array(occ_roi["shape"])

        ellipsoid = get_indices_inside_ellipsoid(
            roi.shape,
            center=np.array(occ_roi["center"]) * roi_zoom,
            semi_axes_lengths=np.array(occ_roi["semi_axes_lengths"]) * roi_zoom,
            euler_angles=occ_roi["euler_angles"],
        )
        roi[~ellipsoid] = 0

        return roi


@requires_field("data_ref")
class TextureAdderHandler(AbstractHandler):
    """Add texture to the image by playing a white noise.

    Parameters
    ----------
    var_texture
        relative factor to compute the noise variance.
    """

    name = "phantom-texture"

    def __init__(self, var_texture: float = 0.001):
        super().__init__()
        self._var_texture = var_texture

    def _handle(self, sim: SimData) -> SimData:
        sigma_noise = self._var_texture * sim.data_ref[0]
        rng = validate_rng(sim.rng)

        sim.data_ref += sigma_noise * rng.standard_normal(
            sim.data_ref.shape[1:], dtype=real_type(sim.data_ref.dtype)
        )

        return sim


@requires_field("data_ref")
@requires_field("data_acq", lambda x: x.data_ref.copy())
@requires_field("roi")
class SlicerHandler(AbstractHandler):
    """Create an handler to get a 2D+T slice from a 3D+T simulation.

    Parameters
    ----------
    axis
        axis position to cut the slice
    index:
        index position on axis where the slice is perform.
    """

    name = "phantom-slicer"

    def __init__(self, axis: int, index: int):
        super().__init__()
        if not (0 <= axis <= 2):
            raise ValueError("only 3D array are supported.")

        self.axis = axis
        self.index = index

    def _run_callback(self, old_sim: SimData, new_sim: SimData) -> None:
        """Notify that we are now in  2D."""
        self.log.warning("Simulation is now 2D")

    @property
    def slicer(self) -> tuple:
        """Return slicer operator."""
        base_slicer: list[slice | int] = [slice(None, None, None)] * 4
        base_slicer[self.axis + 1] = self.index
        return tuple(base_slicer)

    def _handle(self, sim: SimData) -> SimData:
        """Perform the slicing on all relevant data and update data_shape."""
        for data_type in ["data_ref", "data_acq"]:
            if (array := getattr(sim, data_type)) is not None:
                setattr(sim, data_type, array[self.slicer])
        if sim.lazy:
            raise NotImplementedError("Lazy simulation is not supported yet.")
        sim.roi = sim.roi[self.slicer[1:]]  # roi does not have frame dimension.
        new_shape = sim._meta.shape
        sim._meta.shape = tuple(s for ax, s in enumerate(new_shape) if ax != self.axis)
        return sim
