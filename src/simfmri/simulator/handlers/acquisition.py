"""Acquisition Handler to genereate Kspace data from simulation."""
from __future__ import annotations
from typing import Callable
import numpy as np
from fmri.operators.fourier import CartesianSpaceFourier
from simfmri.utils import get_smaps
from simfmri.utils.cartesian_sampling import get_cartesian_mask

from ..simulation import SimulationData
from .base import AbstractHandler


class AcquisitionHandler(AbstractHandler):
    """
    Simulate the acquisition of the data.

    Parameters
    ----------
    sampling_mask
        array or function returning an array of the sampling mask
    gen_smaps
        If true, smaps are also generated, default true.

    """

    def __init__(self, sampling_mask: np.ndarray | Callable, gen_smaps: bool = True):
        super().__init__()
        self._sampling_mask = sampling_mask
        self._gen_smaps = gen_smaps

        pass

    def _handle(self, sim: SimulationData) -> SimulationData:
        if self._gen_smaps and sim.n_coils > 1:
            sim.smaps = get_smaps(sim.shape, sim.n_coils)

        if callable(self._sampling_mask):
            mask = self._sampling_mask(sim)
        else:
            mask = self._sampling_mask
        fourier_op = CartesianSpaceFourier(
            sim.shape,
            mask=mask,
            n_coils=sim.n_coils,
            n_frames=sim.n_frames,
            smaps=sim.smaps,
            n_jobs=-1,
        )

        sim.kspace_data = fourier_op.op(sim.data_acq)
        sim.kspace_mask = mask
        return sim

    @classmethod
    def vds(
        cls,
        acs: float | int,
        accel: int,
        accel_axis: int = -1,
        constant: bool = True,
        gen_smaps: bool = True,
    ) -> AcquisitionHandler:
        """
        Generate Acquisition with Variable density sampling (vds).

        Parameters
        ----------
        acs
            autocalibration line number (int) or proportion (float)
        accel
            Acceleration factor
        constant
            If true, the sampling pattern is repeated across time
        gen_smaps
            If true, smaps are generated, and used in the acquisition.
        """

        def sampling_mask(sim: SimulationData) -> np.ndarray:
            return get_cartesian_mask(
                shape=sim.shape,
                n_frames=sim.n_frames,
                constant=constant,
                center_prop=acs,
                accel=accel,
                accel_axis=accel_axis,
                pdf="gaussian",
                rng=sim._meta.rng,
            )

        return cls(sampling_mask=sampling_mask, gen_smaps=gen_smaps)


class NonCartesianAcquisitionHandler(AbstractHandler):
    """Non Cartesian Acquisition Handler to genereate Kspace data from simulation."""

    def __init__(self, sampling_mask: np.ndarray | Callable, gen_smaps: bool = True):
        super().__init__()
        self._sampling_mask = sampling_mask
        self._gen_smaps = gen_smaps

        pass

    def _handle(self, sim: SimulationData) -> SimulationData:
        if self._gen_smaps:
            sim.smaps = get_smaps(sim.shape, sim.n_coils)
