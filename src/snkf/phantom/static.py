"""Module to create phantom for simulation."""

from __future__ import annotations
import logging
import base64
import contextlib
import os
from dataclasses import dataclass
from enum import IntEnum
from importlib.resources import files
from multiprocessing.managers import SharedMemoryManager
from typing import TypeVar

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray

from snkf.engine.parallel import ArrayProps, run_parallel, array_from_shm, array_to_shm

from ..simulation import SimConfig

log = logging.getLogger(__name__)


class PropTissueEnum(IntEnum):
    """Enum for the tissue properties."""

    T1 = 0
    T2 = 1
    T2s = 2
    rho = 3
    chi = 4


@dataclass
class Phantom:
    """A Phantom consist of a list of tissue mask and parameters for those tissues."""

    name: str
    masks: NDArray[np.float32]
    labels: NDArray[str]
    props: NDArray[np.float32]

    @classmethod
    def from_brainweb(
        cls,
        sub_id: int,
        sim_conf: SimConfig,
        tissue_file: os.PathLike = None,
        tissue_select: list[str] = None,
        tissue_ignore: list[str] = None,
    ) -> Phantom:
        """Get the Brainweb Phantom."""
        from brainweb_dl import get_mri, BrainWebTissuesV2

        from .utils import resize_tissues

        if tissue_ignore is not None and tissue_select is not None:
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
        if tissue_file is None:
            tissue_file = files("snkf.phantom.data") / "tissues_properties.csv"
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
    ) -> Phantom:
        """Give access the tissue masks and properties in shared memory."""
        with array_from_shm(mask_prop, label_prop, properties_prop) as arrs:
            yield cls(name, *arrs)

    def in_shared_memory(
        self, manager: SharedMemoryManager
    ) -> tuple[str, ArrayProps, ArrayProps, ArrayProps]:
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
    def anat_shape(self) -> tuple[int, int, int] | tuple[int, int]:
        """Get the shape of the base volume."""
        return self.masks.shape[1:]

    @property
    def n_tissues(self) -> int:
        """Get the number of tissues."""
        return len(self.masks)


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
