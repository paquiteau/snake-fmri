"""Module to create phantom for simulation."""

from __future__ import annotations
import contextlib
import os
from dataclasses import dataclass
from importlib.resources import files
from typing import TypeVar

import ismrmrd as mrd
from enum import IntEnum
import numpy as np
from numpy.typing import NDArray

from snkf.engine.parallel import run_parallel

from ..simulation import SimConfig


class TissueProperties(IntEnum):
    """Enum for the tissue properties."""

    T1 = 0
    T2 = 1
    T2s = 2
    rho = 3
    chi = 4


@dataclass(frozen=True)
class Phantom:
    """A Phantom consist of a list of tissue mask and parameters for those tissues."""

    name: str
    tissue_masks: NDArray[np.float32]
    tissue_label: NDArray[str]
    tissue_properties: NDArray[np.float32]

    @classmethod
    def from_brainweb(
        cls,
        sub_id: int,
        sim_conf: SimConfig,
        tissue_file: os.PathLike = None,
        tissue_select: list[str] = None,
    ) -> Phantom:
        """Get the Brainweb Phantom."""
        from brainweb_dl import get_mri

        from .utils import resize_tissues

        # TODO: Use the sim shape properly.
        tissues_mask = get_mri(sub_id, contrast="fuzzy").astype(np.float32)
        z = (
            np.array([0.5, 0.5, 0.5])
            * np.array(sim_conf.shape)
            / np.array(sim_conf.fov_mm)
        )
        tissues_mask = np.moveaxis(tissues_mask, -1, 0)
        tissues_list = []
        if tissue_file is None:
            tissue_file = files("snkf.phantom.data") / "tissues_properties.csv"
        with open(tissue_file) as f:
            lines = f.readlines()
            select = []
            for idx, line in enumerate(lines[1:]):
                vals = line.split(",")
                t1, t2, t2s, rho, chi = map(np.float32, vals[1:])
                name = vals[0]
                if not tissue_select or name in tissue_select:
                    t = (name, t1, t2, t2s, rho, chi)
                    tissues_list.append(t)
                    select.append(idx)

        tissues_mask = tissues_mask[select]
        shape = tissues_mask.shape
        new_shape = (shape[0], *np.round(np.array(shape[1:]) * z).astype(int))
        tissue_resized = np.zeros(new_shape, dtype=np.float32)
        run_parallel(
            resize_tissues, tissues_mask, tissue_resized, parallel_axis=0, z=tuple(z)
        )
        tissues_mask = tissue_resized

        return cls(
            "brainweb",
            tissues_mask,
            tissue_label=np.array([t[0] for t in tissues_list]),
            tissue_properties=np.array([t[1:] for t in tissues_list]),
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
        tissue_label = np.array(image.meta["tissue_label"][1:-1].split(" "))
        # FIXME deserialize tissue_properties !
        return cls(
            tissue_masks=image.data,
            tissue_label=tissue_label,
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
            {k: v for k, v in self.__dict__.items() if k != "tissue_masks"}
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
                data=self.tissue_masks,
                attribute_string=meta_sr,
            ),
        )
        return dataset

    @classmethod
    def from_shared_memory() -> Phantom: ...

    @contextlib.contextmanager
    def in_shared_memory() -> tuple:
        """Create a copy of the phantom in shared memory"""

    @property
    def anat_shape(self) -> tuple[int, int, int] | tuple[int, int]:
        """Get the shape of the base volume."""
        return self.tissue_masks.shape[1:]

    @property
    def n_tissues(self) -> int:
        """Get the number of tissues."""
        return len(self.tissue_masks)


T = TypeVar("T")


def get_contrast_gre(
    phantom: Phantom, FA: NDArray, TE: NDArray, TR: NDArray
) -> NDArray:
    """Compute the GRE contrast at TE."""
    return (
        np.sin(FA)
        * np.exp(-TE / phantom.T2s)
        * (1 - np.exp(-TR / phantom.T1))
        / (1 - np.cos(FA) * np.exp(-TR / phantom.T1))
    )
