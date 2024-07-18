"""Acquisition engine using nufft."""

from collections.abc import Generator, Sequence
from copy import deepcopy

import ismrmrd as mrd
import numpy as np
from mrinufft.operators import FourierOperatorBase, get_operator
from numpy.typing import NDArray

from snkf.phantom import DynamicData, Phantom
from snkf.simulation import SimConfig

from .utils import get_contrast_gre

from .base import BaseAcquisitionEngine


class NufftAcquisitionEngine(BaseAcquisitionEngine):
    """Acquisition engine using nufft."""

    def __init__(
        self, mode: str = "T2s", snr: float = 10.0, nufft_backend: str = "gpunufft"
    ):
        super().__init__(mode, snr)
        self.nufft_backend = nufft_backend

    def _job_trajectories(
        self,
        dataset: mrd.Dataset,
        hdr: mrd.xsd.ismrmrdHeader,
        sim_conf: SimConfig,
        shot_idx: Sequence[int],
    ) -> NDArray:
        """Get Non Cartesian trajectories from the dataset.

        Returns
        -------
        NDArray
            The trajectories.
        """
        if not isinstance(shot_idx, Sequence):
            shot_idx = [shot_idx]
        head = dataset._dataset["data"][0]["head"]
        n_samples = head["number_of_samples"]
        ndim = head["trajectory_dimensions"]
        trajectories = np.zeros(len(shot_idx), n_samples, ndim)

        for i, s in enumerate(shot_idx):
            trajectories[i] = dataset._dataset["data"][s]["traj"].reshape(
                n_samples, ndim
            )
        return trajectories

    @staticmethod
    def _init_model_nufft(
        samples: NDArray, sim_conf: SimConfig, smaps: NDArray, backend: str
    ) -> FourierOperatorBase:
        """Initialize the nufft operator."""
        n_coils = len(smaps) if smaps else 1
        kwargs = {}
        if "stacked" in backend:
            kwargs["z_index"] = "auto"

        nufft = get_operator(backend)(
            samples,  # dummy samples locs
            shape=sim_conf.shape,
            n_coils=n_coils,
            smaps=smaps,
            density=False,
            **kwargs,
        )
        return nufft

    @staticmethod
    def _job_model_T2s(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: NDArray,
        smaps: NDArray,
        backend: str,
        echo_sample_idx: int,
    ) -> np.ndarray:
        """Acquire k-space data with T2s relaxation effect."""
        chunk_size, n_samples, ndim = trajectories.shape

        final_ksp = np.zeros(
            (chunk_size, sim_conf.hardware.n_coils, n_samples), dtype=np.complex64
        )
        # (n_tissues_true, n_samples) Filter the tissues that have NaN Values.
        nufft = NufftAcquisitionEngine._init_model_nufft(
            trajectories[0], sim_conf, smaps, backend=backend
        )
        echo_idx = np.argmin(np.sum(np.abs(trajectories[0]) ** 2), axis=1)

        t2s_decay = BaseAcquisitionEngine._job_get_T2s_decay(
            sim_conf.hardware.dwell_time_ms, echo_idx, n_samples, phantom
        )
        for i, traj in enumerate(trajectories):
            frame_phantom = deepcopy(phantom)
            for dyn_data in dyn_datas:
                frame_phantom = dyn_data.func(frame_phantom, dyn_data.data, i)
            # Apply the contrast tissue-wise
            contrast = get_contrast_gre(
                frame_phantom,
                sim_conf.seq.FA,
                sim_conf.seq.TE,
                sim_conf.seq.TR,
            )
            phantom_state = (
                1000
                * contrast[(..., *([None] * len(frame_phantom.anat_shape)))]
                * frame_phantom.masks
            )
            nufft.samples = traj
            nufft.n_batchs = len(phantom_state)
            ksp = nufft.op(phantom_state)
            # apply the T2s and sum over tissues
            # final_ksp[i] = np.sum(ksp * t2s_decay[:, None, :], axis=0)
            final_ksp[i] = np.einsum("kij, kj-> ij", ksp, t2s_decay)
        return final_ksp

    @staticmethod
    def _job_model_simple(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: Generator[NDArray],
        nufft: FourierOperatorBase,
        backend: str,
        echo_sample_idx: int,
    ) -> np.ndarray:
        """Acquire k-space data. No T2s decay."""
        chunk_size, n_samples, ndim = trajectories.shape

        final_ksp = np.zeros(
            (chunk_size, sim_conf.hardware.n_coils, n_samples), dtype=np.complex64
        )
        # (n_tissues_true, n_samples) Filter the tissues that have NaN Values
        for i, traj in enumerate(trajectories):
            frame_phantom = deepcopy(phantom)
            for dyn_data in dyn_datas:
                frame_phantom = dyn_data.func(frame_phantom, dyn_data.data, i)
            # reduced the array, we dont have batch tissues !
            contrast = get_contrast_gre(
                frame_phantom,
                sim_conf.seq.FA,
                sim_conf.seq.TE,
                sim_conf.seq.TR,
            )
            phantom_state = np.sum(
                1000
                * contrast[(..., *([None] * len(phantom.anat_shape)))]
                * frame_phantom.masks,
                axis=0,
            )
            nufft.samples = traj
            final_ksp[i] = nufft.op(phantom_state)
        return final_ksp

    def _write_chunk_data(
        self, dataset: mrd.Dataset, chunk: Sequence[int], chunk_data: NDArray
    ) -> None:
        for i, shot in enumerate(chunk):
            acq = dataset.read_acquisition(shot)
            acq.data[:] = chunk_data[i]
            dataset.write_acquisition(acq, shot)
