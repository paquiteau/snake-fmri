"""Module to create phantom for simulation."""

from __future__ import annotations
from numpy.typing import NDArray
import numpy as np
import os
from importlib.resources import files
from dataclasses import dataclass
from typing import TypeVar
from snkf.engine.parallel import run_parallel


@dataclass(frozen=True)
class Phantom:
    """A Phantom consist of a list of tissue mask and parameters for those tissues."""

    name: str
    tissue_masks: NDArray[np.float32]
    tissue_label: NDArray[str]
    T1: NDArray[np.float32]
    T2: NDArray[np.float32]
    T2s: NDArray[np.float32]
    rho: NDArray[np.float32]
    chi: NDArray[np.float32]

    @classmethod
    def from_brainweb(
        cls,
        sub_id: int,
        resolution_mm: float,
        tissue_file: os.PathLike = None,
        tissue_select: list[str] = None,
    ) -> Phantom:
        """Get the Brainweb Phantom."""
        from brainweb_dl import get_mri
        from .utils import resize_tissues

        tissues_mask = get_mri(sub_id, contrast="fuzzy").astype(np.float32)
        z = np.array([0.5, 0.5, 0.5]) / np.array(resolution_mm)
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
            *[np.array(prop) for prop in zip(*tissues_list, strict=False)],
        )

    @classmethod
    def from_shepp_logan(cls, resolution: tuple[int]) -> Phantom:
        """Get the Shepp-Logan Phantom."""
        raise NotImplementedError

    @classmethod
    def from_guerin_kern(cls, resolution: tuple[int]) -> Phantom:
        """Get the Guerin-Kern Phantom."""
        raise NotImplementedError


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
