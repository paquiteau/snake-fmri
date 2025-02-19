"""Module to create phantom for simulation."""

from __future__ import annotations
from copy import deepcopy

import base64
import contextlib
import logging
import os
from collections.abc import Generator
from dataclasses import dataclass, field
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Literal, TypeVar, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import GenericPath

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray
from nibabel.nifti1 import Nifti1Image

from ..parallel import ArrayProps, array_from_shm, array_to_shm
from snake._meta import ThreeFloats
from ..simulation import SimConfig
from snake.core.parallel import run_parallel
from snake.core.phantom.utils import resize_tissues
from snake.core.smaps import get_smaps
from .contrast import _contrast_gre
from .utils import PropTissueEnum, TissueFile

log = logging.getLogger(__name__)


@dataclass
class Phantom:
    """A Phantom consist of all spatial maps that are used in the simulation.

    It is a dataclass that contains the tissue masks, properties, labels, and
    spatial maps of the phantom.

    The tissue masks are a 3D array with the shape (n_tissues, x, y, z), where
    n_tissues is the number of tissues.

    The properties are a 2D array with the shape (n_tissues, n_properties), where
    n_properties is the number of properties.

    The labels are a 1D array with the shape (n_tissues,) containing the names
    of the tissues.

    The sensitivity maps are a 4D array with the shape (n_coils, x, y, z), where
    n_coils is the number of coils.

    The affine matrix is a 2D array with the shape (4, 4) containing the affine
    transformation matrix.

    """

    # TODO Add field map inhomogeneity in the phantom
    name: str
    masks: NDArray[np.float32]
    labels: NDArray[np.string_]
    props: NDArray[np.float32]
    smaps: NDArray[np.complex64] | None = None
    affine: NDArray[np.float32] = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )

    def add_tissue(
        self,
        tissue_name: str,
        mask: NDArray[np.float32],
        props: NDArray[np.float32],
        phantom_name: str | None = None,
    ) -> Phantom:
        """Add a tissue to the phantom. Creates a new Phantom object."""
        masks = np.concatenate((self.masks, mask[None, ...]), axis=0)
        labels = np.concatenate((self.labels, np.array([tissue_name])))
        props = np.concatenate((self.props, props), axis=0)
        return Phantom(
            phantom_name or self.name, masks, labels, props, smaps=self.smaps
        )

    @property
    def labels_idx(self) -> dict[str, int]:
        """Get the index of the labels."""
        return {label: i for i, label in enumerate(self.labels)}

    def make_smaps(
        self, n_coils: int = None, sim_conf: SimConfig = None, antenna: str = "birdcage"
    ) -> None:
        """Get coil sensitivity maps for the phantom."""
        if n_coils is None and sim_conf is not None:
            n_coils = sim_conf.hardware.n_coils
        elif sim_conf is None and n_coils is None:
            raise ValueError("Either n_coils or sim_conf must be provided.")
        if n_coils == 1:
            log.warning("Only one coil, no need for smaps.")
        elif n_coils > 1 and self.smaps is None:
            self.smaps = get_smaps(self.anat_shape, n_coils=n_coils, antenna=antenna)
            log.debug(f"Created smaps for {n_coils} coils.")
        elif self.smaps is not None:
            log.warning("Smaps already exists.")

    @classmethod
    def from_brainweb(
        cls,
        sub_id: int,
        sim_conf: SimConfig,
        tissue_file: str | TissueFile = TissueFile.tissue_1T5,
        tissue_select: list[str] | None = None,
        tissue_ignore: list[str] | None = None,
        output_res: float | ThreeFloats = 0.5,
    ) -> Phantom:
        """Get the Brainweb Phantom.

        Parameters
        ----------
        sub_id: int
            Subject ID of the brainweb dataset.
        sim_conf: SimConfig
            Simulation configuration.
        tissue_file: str
            File with the tissue properties.
        tissue_select: list[str]
            List of tissues to select.
        tissue_ignore: list[str]
            List of tissues to ignore.
        output_res: float
            Resolution of the output phantom.

        Returns
        -------
        Phantom
            The phantom object.
        """
        from brainweb_dl import BrainWebTissuesV2, get_mri

        if tissue_ignore and tissue_select:
            raise ValueError("Only one of tissue_select or tissue_ignore can be used.")
        if tissue_select:
            tissue_select = [t.lower() for t in tissue_select]
        if tissue_ignore:
            tissue_ignore = [t.lower() for t in tissue_ignore]

        tissues_mask, affine = get_mri(sub_id, contrast="fuzzy", with_affine=True)
        tissues_mask = tissues_mask.astype(np.float32)
        affine = affine.astype(np.float32)
        tissues_mask = np.ascontiguousarray(tissues_mask.T)
        tissues_list = []
        try:
            if isinstance(tissue_file, TissueFile):
                tissue_file = tissue_file.value
            else:
                tissue_file = TissueFile[tissue_file].value
        except ValueError as exc:
            if not os.path.exists(tissue_file):
                raise FileNotFoundError(f"File {tissue_file} does not exist.") from exc
        finally:
            tissue_file = str(tissue_file)
        log.info(f"Using tissue file:{tissue_file} ")
        with open(tissue_file) as f:
            lines = f.readlines()
            select = []
            for line in lines[1:]:
                vals = line.split(",")
                t1, t2, t2s, rho, chi = map(np.float32, vals[1:])
                name = vals[0]
                t = (name, t1, t2, t2s, rho, chi)
                if (
                    (tissue_select and name in tissue_select)
                    or (tissue_ignore and name not in tissue_ignore)
                    or (not tissue_select and not tissue_ignore)
                ):
                    tissues_list.append(t)
                    select.append(BrainWebTissuesV2[name.upper()])
        log.info(
            f"Selected tissues: {select}, {[t[0] for t in tissues_list]}",
        )
        if len(tissues_list) == 0:
            raise ValueError("No tissues selected")
        tissues_mask = tissues_mask[select]
        shape = tissues_mask.shape
        #
        # TODO: Use the sim shape properly.
        if output_res != 0.5:
            if isinstance(output_res, float):
                output_res = [output_res] * 3
            z = np.array([0.5, 0.5, 0.5]) / np.array(output_res)
            new_shape = (shape[0], *np.round(np.array(shape[1:]) * z).astype(int))
            tissue_resized = np.zeros(new_shape, dtype=np.float32)

            run_parallel(
                resize_tissues,
                tissues_mask,
                tissue_resized,
                parallel_axis=0,
                z=tuple(z),
            )
            tissues_mask = tissue_resized

        smaps = None
        if sim_conf.hardware.n_coils > 1:
            smaps = get_smaps(
                tissues_mask.shape[1:],
                n_coils=sim_conf.hardware.n_coils,
            )

        return cls(
            "brainweb-{sub_id:02d}",
            tissues_mask,
            labels=np.array([t[0] for t in tissues_list]),
            props=np.array([t[1:] for t in tissues_list]),
            smaps=smaps,
            affine=affine,
        )

    @classmethod
    def from_mrd_dataset(
        cls, dataset: mrd.Dataset | os.PathLike, imnum: int = 0
    ) -> Phantom:
        """Load the phantom from a mrd dataset."""
        if not isinstance(dataset, mrd.Dataset):
            dataset = mrd.Dataset(dataset, create_if_needed=False)
        image = dataset.read_image("phantom", imnum)
        name = image.meta.pop("name")
        labels = np.array(image.meta["labels"].split(","))
        props = unserialize_array(image.meta["props"])

        # Affine matrix from the header
        position = image._head.position
        read_dir = image._head.read_dir
        phase_dir = image._head.phase_dir
        slice_dir = image._head.slice_dir
        affine = np.eye(4, dtype=np.float32)
        res = np.array(image._head.field_of_view) / np.array(image._head.matrix_size)
        print("res", tuple(image._head.field_of_view))
        affine[:3, 3] = -position[0], -position[1], position[2]
        affine[:3, 0] = (
            -read_dir[0] * res[0],
            -read_dir[1] * res[0],
            read_dir[2] * res[0],
        )
        affine[:3, 1] = (
            -phase_dir[0] * res[1],
            -phase_dir[1] * res[1],
            phase_dir[2] * res[1],
        )
        affine[:3, 2] = (
            -slice_dir[0] * res[2],
            -slice_dir[1] * res[2],
            slice_dir[2] * res[2],
        )

        # smaps
        try:
            smaps = dataset.read_image("smaps", imnum).data
        except LookupError:
            smaps = None

        return cls(
            masks=image.data,
            labels=labels,
            props=props,
            name=name,
            affine=affine,
            smaps=smaps,
        )

    def to_mrd_dataset(self, dataset: mrd.Dataset | GenericPath) -> mrd.Dataset:
        """Add the phantom as an image to the dataset."""
        # Create the image
        if not isinstance(dataset, mrd.Dataset):
            dataset = mrd.Dataset(dataset, create_if_needed=True)

        meta_sr = mrd.Meta(
            {
                "name": self.name,
                "labels": f"{','.join(self.labels)}",
                "props": serialize_array(self.props),
                "affine": serialize_array(self.affine),
            }
        ).serialize()

        # Convert the affine matrix to position, field of view, etc.
        offsets = self.affine[:3, 3]
        position = (-offsets[0], -offsets[1], offsets[2])
        read_dir = self.affine[:3, 0] / self.affine[0, 0]
        read_dir = (-read_dir[0], -read_dir[1], read_dir[2])
        phase_dir = self.affine[:3, 1] / self.affine[1, 1]
        phase_dir = (-phase_dir[0], -phase_dir[1], phase_dir[2])
        slice_dir = self.affine[:3, 2] / self.affine[2, 2]
        slice_dir = (-slice_dir[0], -slice_dir[1], slice_dir[2])

        fov_mm = tuple(np.float32(np.array(self.anat_shape) * np.diag(self.affine)[:3]))

        # Add the phantom data
        dataset.append_image(
            "phantom",
            mrd.image.Image(
                head=mrd.image.ImageHeader(
                    matrix_size=self.anat_shape,
                    field_of_view=fov_mm,
                    position=position,
                    phase_dir=phase_dir,
                    slice_dir=slice_dir,
                    read_dir=read_dir,
                    channels=self.n_tissues,
                    acquisition_time_stamp=0,
                    attribute_string_len=len(meta_sr),
                ),
                data=self.masks,
                attribute_string=meta_sr,
            ),
        )
        # Add the smaps
        if self.smaps is not None:
            dataset.append_image(
                "smaps",
                mrd.image.Image(
                    head=mrd.image.ImageHeader(
                        matrix_size=self.anat_shape,
                        field_of_view=fov_mm,
                        position=position,
                        phase_dir=phase_dir,
                        slice_dir=slice_dir,
                        read_dir=read_dir,
                        channels=len(self.smaps),
                        acquisition_time_stamp=0,
                    ),
                    data=self.smaps,
                ),
            )
        return dataset

    @classmethod
    @contextlib.contextmanager
    def from_shared_memory(
        cls,
        name: str,
        mask_prop: ArrayProps,
        properties_prop: ArrayProps,
        label_prop: ArrayProps,
        smaps_prop: ArrayProps,
        affine_prop: ArrayProps,
    ) -> Generator[Phantom, None, None]:
        """Give access the tissue masks and properties in shared memory."""
        with array_from_shm(
            mask_prop, label_prop, properties_prop, smaps_prop, affine_prop
        ) as arrs:
            yield cls(name, *arrs)

    def in_shared_memory(
        self, manager: SharedMemoryManager
    ) -> tuple[
        tuple[str, ArrayProps, ArrayProps, ArrayProps, ArrayProps | None, ArrayProps],
        tuple[
            SharedMemory, SharedMemory, SharedMemory, SharedMemory | None, SharedMemory
        ],
    ]:
        """Add a copy of the phantom in shared memory."""
        tissue_mask, _, tisue_mask_smm = array_to_shm(self.masks, manager)
        tissue_props, _, tissue_prop_smm = array_to_shm(self.props, manager)
        labels, _, labels_sm = array_to_shm(self.labels, manager)
        affine, _, affine_sm = array_to_shm(self.affine, manager)
        if self.smaps is not None:
            smaps, _, smaps_sm = array_to_shm(self.smaps, manager)
        else:
            smaps, smaps_sm = None, None

        return (
            (self.name, tissue_mask, tissue_props, labels, smaps, affine),
            (tisue_mask_smm, tissue_prop_smm, labels_sm, smaps_sm, affine_sm),
        )

    def masks2nifti(self) -> Nifti1Image:
        """Return the masks of the phantom as a Nifti object."""
        return Nifti1Image(
            self.masks,
            affine=self.affine,
            extra={"props": self.props, "labels": self.labels},
        )

    def smaps2nifti(self) -> Nifti1Image:
        """Return the smaps as a Nifti object."""
        if self.smaps:
            return Nifti1Image(self.smaps, affine=self.affine)
        else:
            raise ValueError("No Smaps to convert.")

    def contrast(
        self,
        *,
        TR: float | None = None,
        TE: float | None = None,
        FA: float | None = None,
        sequence: Literal["GRE"] = "GRE",
        sim_conf: SimConfig | None = None,
        aggregate: bool = True,
    ) -> NDArray[np.float32]:
        """Compute the contrast of the phantom for a given sequence.

        Parameters
        ----------
        TR: float
        TE: float
        FA: float
        sim_conf: SimConfig
            Other way to provide sequence parameters
        aggregate: bool, optional default=True
            Sum all the tissues contrast for getting a single image.
        sequence="GRE"
            Default value, no other value is currently supported.

        Results
        -------
        NDArray
            The constrast of the tissues.
        """
        if sim_conf is not None:
            TR = sim_conf.seq.TR_eff  # Here we use the effective TR.
            TE = sim_conf.seq.TE
            FA = sim_conf.seq.FA
        if sim_conf is None and TR is None and TE is None and FA is None:
            raise ValueError("Missing either sim_conf or TR,TE,FA")
        if sequence.upper() == "GRE":
            contrasts = _contrast_gre(self.props, TR=TR, TE=TE, FA=FA)
        else:
            raise NotImplementedError("Contrast not implemented.")
        if aggregate:
            ret = np.zeros(self.anat_shape, dtype=np.float32)
            for c, m in zip(contrasts, self.masks, strict=False):
                ret += c * m
            return ret
        else:
            return self.masks * contrasts[(..., *([None] * len(self.anat_shape)))]

    @property
    def anat_shape(self) -> tuple[int, ...]:
        """Get the shape of the base volume."""
        return self.masks.shape[1:]

    @property
    def n_tissues(self) -> int:
        """Get the number of tissues."""
        return len(self.masks)

    def __repr__(self):
        ret = f"Phantom[{self.name}]: {self.props.shape}\n"
        ret += f"{'tissue name':14s}" + "".join(
            f"{str(prop):4s}" for prop in PropTissueEnum
        )
        ret += "\n"
        for i, tissue_name in enumerate(self.labels):
            props = self.props[i]
            ret += (
                f"{tissue_name:14s}"
                + "".join(f"{props[p]:4}" for p in PropTissueEnum.__members__.values())
                + "\n"
            )
        return ret

    def __deepcopy__(self, memo: Any) -> Phantom:
        """Create a copy of the phantom."""
        return Phantom(
            name=self.name,
            masks=deepcopy(self.masks, memo),
            labels=deepcopy(self.labels, memo),
            props=deepcopy(self.props, memo),
            smaps=deepcopy(self.smaps, memo),
        )

    def copy(self) -> Phantom:
        """Return deep copy of the Phantom."""
        return deepcopy(self)


T = TypeVar("T")


def serialize_array(arr: NDArray) -> str:
    """Serialize the array for mrd compatible format."""
    return "__".join(
        [
            base64.b64encode(arr.tobytes()).decode(),
            str(arr.shape),
            str(arr.dtype),
        ]
    )


def unserialize_array(s: str) -> NDArray:
    """Unserialize the array for mrd compatible format."""
    data, shape, dtype = s.split("__")
    shape = eval(shape)  # FIXME
    return np.frombuffer(base64.b64decode(data.encode()), dtype=dtype).reshape(*shape)
