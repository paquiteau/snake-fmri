"""Module to create phantom for simulation."""

from __future__ import annotations

import base64
import contextlib
import logging
import os
from collections.abc import Generator
from dataclasses import dataclass
from enum import IntEnum
from importlib.resources import files
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import TypeVar

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray

from ..parallel import ArrayProps, array_from_shm, array_to_shm, run_parallel
from snake._meta import NoCaseEnum
from ..simulation import SimConfig

log = logging.getLogger(__name__)


class PropTissueEnum(IntEnum):
    """Enum for the tissue properties."""

    T1 = 0
    T2 = 1
    T2s = 2
    rho = 3
    chi = 4


class TissueFile(str, NoCaseEnum):
    """Enum for the tissue properties file."""

    tissue_1T5 = str(files("snake.core.phantom.data") / "tissues_properties_1T5.csv")
    tissue_7T = str(files("snake.core.phantom.data") / "tissues_properties_7T.csv")


@dataclass
class Phantom:
    """A Phantom consist of a list of tissue mask and parameters for those tissues."""

    name: str
    masks: NDArray[np.float32]
    labels: NDArray[np.string_]
    props: NDArray[np.float32]

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
        return Phantom(phantom_name or self.name, masks, labels, props)

    @classmethod
    def from_brainweb(
        cls,
        sub_id: int,
        sim_conf: SimConfig,
        tissue_file: str | TissueFile = TissueFile.tissue_1T5,
        tissue_select: list[str] | None = None,
        tissue_ignore: list[str] | None = None,
    ) -> Phantom:
        """Get the Brainweb Phantom."""
        from brainweb_dl import BrainWebTissuesV2, get_mri

        from .utils import resize_tissues

        if tissue_ignore and tissue_select:
            raise ValueError("Only one of tissue_select or tissue_ignore can be used.")

        # TODO: Use the sim shape properly.
        tissues_mask = get_mri(sub_id, contrast="fuzzy").astype(np.float32)
        z = (
            np.array([0.5, 0.5, 0.5])
            * np.array(sim_conf.shape)
            / np.array(sim_conf.fov_mm)
        )
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
        tissues_mask = tissues_mask[select]
        shape = tissues_mask.shape
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

        return cls(
            "brainweb",
            tissues_mask,
            labels=np.array([t[0] for t in tissues_list]),
            props=np.array([t[1:] for t in tissues_list]),
        )

    @classmethod
    def from_shepp_logan(cls, resolution: tuple[int]) -> Phantom:
        """Get the Shepp-Logan Phantom."""
        raise NotImplementedError

    @classmethod
    def from_guerin_kern(cls, resolution: tuple[int]) -> Phantom:
        """Get the Guerin-Kern Phantom."""
        raise NotImplementedError

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

        return cls(
            masks=image.data,
            labels=labels,
            props=props,
            name=name,
        )

    def to_mrd_dataset(
        self, dataset: mrd.Dataset, sim_conf: SimConfig, imnum: int = 0
    ) -> mrd.Dataset:
        """Add the phantom as an image to the dataset."""
        # Create the image
        if not isinstance(dataset, mrd.Dataset):
            dataset = mrd.Dataset(dataset, create_if_needed=True)

        meta_sr = mrd.Meta(
            {
                "name": self.name,
                "labels": f'{",".join(self.labels)}',
                "props": serialize_array(self.props),
            }
        ).serialize()

        dataset.append_image(
            "phantom",
            mrd.image.Image(
                head=mrd.image.ImageHeader(
                    matrixSize=mrd.xsd.matrixSizeType(*self.anat_shape),
                    fieldOfView_mm=mrd.xsd.fieldOfViewMm(*sim_conf.fov_mm),
                    channels=self.n_tissues,
                    acquisition_time_stamp=0,
                    attribute_string_len=len(meta_sr),
                ),
                data=self.masks,
                attribute_string=meta_sr,
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
    ) -> Generator[Phantom, None, None]:
        """Give access the tissue masks and properties in shared memory."""
        with array_from_shm(mask_prop, label_prop, properties_prop) as arrs:
            yield cls(name, *arrs)

    def in_shared_memory(self, manager: SharedMemoryManager) -> tuple[
        tuple[str, ArrayProps, ArrayProps, ArrayProps],
        tuple[SharedMemory, SharedMemory, SharedMemory],
    ]:
        """Add a copy of the phantom in shared memory."""
        tissue_mask, _, tisue_mask_smm = array_to_shm(self.masks, manager)
        tissue_props, _, tissue_prop_smm = array_to_shm(self.props, manager)
        labels, _, labels_sm = array_to_shm(self.labels, manager)

        return (
            (self.name, tissue_mask, tissue_props, labels),
            (
                tisue_mask_smm,
                tissue_prop_smm,
                labels_sm,
            ),
        )

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
