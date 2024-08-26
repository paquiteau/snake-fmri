"""Acquisition engine for Cartesian trajectories."""

from collections.abc import Sequence
from copy import deepcopy

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray

from snake.core.phantom import DynamicData, Phantom
from snake.core.simulation import SimConfig
from snake.mrd_utils import MRDLoader

from .base import BaseAcquisitionEngine
from .utils import fft, get_contrast_gre


class EPIAcquisitionEngine(BaseAcquisitionEngine):
    """Acquisition engine for EPI base trajectories."""

    __engine_name__ = "EPI"
    model: str = "simple"
    snr: float = np.inf

    def _get_chunk_list(self, data_loader: MRDLoader) -> Sequence[int]:
        limits = data_loader.header.encoding[0].encodingLimits
        self.n_lines_epi = limits.kspace_encoding_step_1.maximum

        n_epi = data_loader.n_acquisition // self.n_lines_epi
        return range(n_epi)

    def _job_trajectories(
        self,
        data_loader: MRDLoader,
        hdr: mrd.xsd.ismrmrdHeader,
        sim_conf: SimConfig,
        chunk: Sequence[int],
    ) -> np.ndarray:
        """Generate the fourier operator by iterating the dataset."""
        if not isinstance(chunk, Sequence):
            chunk = [chunk]

        limits = hdr.encoding[0].encodingLimits
        n_lines_epi = limits.kspace_encoding_step_1.maximum
        readout_length = limits.kspace_encoding_step_0.maximum
        # Read all the chunk data from file.
        traj = (
            data_loader._dataset["data"][
                chunk[0] * n_lines_epi : (chunk[-1] + 1) * n_lines_epi
            ]["traj"]
            .astype(np.int32, copy=False)
            .reshape(len(chunk), n_lines_epi, readout_length, 3)
        )

        return traj

    @staticmethod
    def _job_model_T2s(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: NDArray,  # (Chunksize, N, 3)
        smaps: NDArray,
    ) -> np.ndarray:
        """Acquire k-space data. With T2s decay."""
        readout_length = trajectories.shape[-2]
        n_lines_epi = trajectories.shape[-3]

        final_ksp = np.zeros(
            (
                len(trajectories),
                sim_conf.hardware.n_coils,
                n_lines_epi,
                readout_length,
            ),
            dtype=np.complex64,
        )

        n_samples = int(readout_length * n_lines_epi)
        shape = sim_conf.shape
        echo_idx = int(
            np.argmin(
                np.sum(
                    abs(
                        trajectories[0].reshape(-1, 3)
                        - (shape[0] // 2, shape[1] // 2, shape[2] // 2)
                    )
                    ** 2,
                    axis=-1,
                )
            )
        )

        t2s_decay = BaseAcquisitionEngine._job_get_T2s_decay(
            sim_conf.hardware.dwell_time_ms, echo_idx, n_samples, phantom
        )

        for i, epi_2d in enumerate(trajectories):
            frame_phantom = deepcopy(phantom)
            for dyn_data in dyn_datas:
                frame_phantom = dyn_data.func(frame_phantom, dyn_data.data, i)

            contrast = get_contrast_gre(
                frame_phantom,
                sim_conf.seq.FA,
                sim_conf.seq.TE,
                sim_conf.seq.TR,
            )
            phantom_state = (
                contrast[(..., *([None] * len(frame_phantom.anat_shape)))]
                * frame_phantom.masks
            )

            if smaps is None:
                ksp = fft(phantom_state[:, None, ...], axis=(-3, -2, -1))
            else:
                ksp = fft(phantom_state[:, None, ...] * smaps, axis=(-3, -2, -1))
            flat_epi = epi_2d.reshape(-1, 3)
            for c in range(sim_conf.hardware.n_coils):
                ksp_coil_sum = np.zeros(
                    (n_lines_epi * readout_length), dtype=np.complex64
                )
                for b in range(phantom_state.shape[0]):
                    ksp_coil_sum += ksp[b, c][tuple(flat_epi.T)] * t2s_decay[b]
                final_ksp[i, c] = ksp_coil_sum.reshape((n_lines_epi, readout_length))

        return final_ksp

    @staticmethod
    def _job_model_simple(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: NDArray,  # (Chunksize, N, 3)
        smaps: NDArray,
    ) -> np.ndarray:
        """Acquire k-space data. No T2s decay."""
        final_ksp = np.zeros(
            (
                len(trajectories),
                sim_conf.hardware.n_coils,
                trajectories.shape[-3],
                trajectories.shape[-2],
            ),
            dtype=np.complex64,
        )
        for i, epi_2d in enumerate(trajectories):
            frame_phantom = deepcopy(phantom)
            for dyn_data in dyn_datas:
                frame_phantom = dyn_data.func(frame_phantom, dyn_data.data, i)

            # Reduce the array, we dont have batch tissues !
            contrast = get_contrast_gre(
                frame_phantom,
                sim_conf.seq.FA,
                sim_conf.seq.TE,
                sim_conf.seq.TR,
            )
            phantom_state = np.sum(
                contrast[(..., *([None] * len(phantom.anat_shape)))]
                * frame_phantom.masks,
                axis=0,
            )
            if smaps is None:
                ksp = fft(phantom_state[None, ...], axis=(-3, -2, -1))
            else:
                ksp = fft(phantom_state[None, ...] * smaps, axis=(-3, -2, -1))
            flat_epi = epi_2d.reshape(-1, 3)
            for c in range(sim_conf.hardware.n_coils):
                ksp_coil = ksp[c]
                a = ksp_coil[tuple(flat_epi.T)]
                final_ksp[i, c] = a.reshape(
                    trajectories.shape[-3],
                    trajectories.shape[-2],
                )
        return final_ksp

    def _write_chunk_data(
        self, data_loader: MRDLoader, chunk: Sequence[int], chunk_data: NDArray
    ) -> None:
        shots = np.concatenate(
            [
                np.arange(
                    shot * self.n_lines_epi,
                    (shot + 1) * self.n_lines_epi,
                    dtype=np.int32,
                )
                for shot in chunk
            ]
        )

        chunk_data = chunk_data.view(np.float32)
        chunk_data = np.moveaxis(
            chunk_data, 1, 2
        )  # put the coil axis after the readout axis
        acq_chunk = data_loader._dataset["data"][shots]
        acq_chunk["data"] = chunk_data.reshape(acq_chunk["data"].shape)
        data_loader._dataset["data"][shots] = acq_chunk


class EVIAcquisition(EPIAcquisitionEngine):
    """EVI Acquisition engine. Same as EPI, but the shots are longer."""

    __engine_name__ = "EVI"
    model: str = "simple"
    snr: float = np.inf

    def _get_chunk_list(self, data_loader: MRDLoader) -> Sequence[int]:
        limits = data_loader.header.encoding[0].encodingLimits
        self.n_lines_epi = limits.kspace_encoding_step_1.maximum
        self.n_slice_epi = limits.slice.maximum
        n_evi = data_loader.n_acquisition // (self.n_lines_epi * self.n_slice_epi)
        return range(n_evi)

    def _job_trajectories(
        self,
        data_loader: MRDLoader,
        hdr: mrd.xsd.ismrmrdHeader,
        sim_conf: SimConfig,
        chunk: Sequence[int],
    ) -> np.ndarray:
        """Generate the fourier operator by iterating the dataset."""
        if not isinstance(chunk, Sequence):
            chunk = [chunk]

        limits = hdr.encoding[0].encodingLimits
        n_lines_epi = limits.kspace_encoding_step_1.maximum
        readout_length = limits.kspace_encoding_step_0.maximum
        slice = limits.slice.maximum
        # Read all the chunk data from file.
        traj = (
            data_loader._dataset["data"][
                chunk[0] * n_lines_epi * slice : (chunk[-1] + 1) * n_lines_epi * slice
            ]["traj"]
            .astype(np.int32, copy=False)
            .reshape(len(chunk), slice, n_lines_epi, readout_length, 3)
        )

        return traj

    @staticmethod
    def _job_model_T2s(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: NDArray,  # (Chunksize, N, 3)
        smaps: NDArray,
    ) -> np.ndarray:
        """Acquire k-space data. With T2s decay."""
        readout_length = trajectories.shape[-2]
        n_lines_epi = trajectories.shape[-3]
        n_slice = trajectories.shape[-4]
        final_ksp = np.zeros(
            (
                len(trajectories),
                sim_conf.hardware.n_coils,
                n_slice,
                n_lines_epi,
                readout_length,
            ),
            dtype=np.complex64,
        )

        n_samples = int(readout_length * n_lines_epi * n_slice)
        shape = sim_conf.shape
        echo_idx = int(
            np.argmin(
                np.sum(
                    abs(
                        trajectories[0].reshape(-1, 3)
                        - (shape[0] // 2, shape[1] // 2, shape[2] // 2)
                    )
                    ** 2,
                    axis=-1,
                )
            )
        )

        t2s_decay = BaseAcquisitionEngine._job_get_T2s_decay(
            sim_conf.hardware.dwell_time_ms, echo_idx, n_samples, phantom
        )

        for i, evi in enumerate(trajectories):
            frame_phantom = deepcopy(phantom)
            for dyn_data in dyn_datas:
                frame_phantom = dyn_data.func(frame_phantom, dyn_data.data, i)

            contrast = get_contrast_gre(
                frame_phantom,
                sim_conf.seq.FA,
                sim_conf.seq.TE,
                sim_conf.seq.TR,
            )
            phantom_state = (
                contrast[(..., *([None] * len(frame_phantom.anat_shape)))]
                * frame_phantom.masks
            )

            if smaps is None:
                ksp = fft(phantom_state[:, None, ...], axis=(-3, -2, -1))
            else:
                ksp = fft(phantom_state[:, None, ...] * smaps, axis=(-3, -2, -1))
            flat_evi = evi.reshape(-1, 3)
            for c in range(sim_conf.hardware.n_coils):
                ksp_coil_sum = np.zeros(
                    (n_lines_epi * readout_length * n_slice), dtype=np.complex64
                )
                for b in range(phantom_state.shape[0]):
                    ksp_coil_sum += ksp[b, c][tuple(flat_evi.T)] * t2s_decay[b]
                final_ksp[i, c] = ksp_coil_sum.reshape(
                    (n_slice, n_lines_epi, readout_length)
                )

        return final_ksp

    @staticmethod
    def _job_model_simple(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: NDArray,  # (Chunksize, N, 3)
        smaps: NDArray,
    ) -> np.ndarray:
        """Acquire k-space data. No T2s decay."""
        final_ksp = np.zeros(
            (
                len(trajectories),
                sim_conf.hardware.n_coils,
                trajectories.shape[-3],
                trajectories.shape[-2],
            ),
            dtype=np.complex64,
        )
        for i, epi_2d in enumerate(trajectories):
            frame_phantom = deepcopy(phantom)
            for dyn_data in dyn_datas:
                frame_phantom = dyn_data.func(frame_phantom, dyn_data.data, i)

            # Reduce the array, we dont have batch tissues !
            contrast = get_contrast_gre(
                frame_phantom,
                sim_conf.seq.FA,
                sim_conf.seq.TE,
                sim_conf.seq.TR,
            )
            phantom_state = np.sum(
                contrast[(..., *([None] * len(phantom.anat_shape)))]
                * frame_phantom.masks,
                axis=0,
            )
            if smaps is None:
                ksp = fft(phantom_state[None, ...], axis=(-3, -2, -1))
            else:
                ksp = fft(phantom_state[None, ...] * smaps, axis=(-3, -2, -1))
            flat_epi = epi_2d.reshape(-1, 3)
            for c in range(sim_conf.hardware.n_coils):
                ksp_coil = ksp[c]
                a = ksp_coil[tuple(flat_epi.T)]
                final_ksp[i, c] = a.reshape(
                    trajectories.shape[-3],
                    trajectories.shape[-2],
                )
        return final_ksp

    def _write_chunk_data(
        self, data_loader: MRDLoader, chunk: Sequence[int], chunk_data: NDArray
    ) -> None:
        shots = np.concatenate(
            [
                np.arange(
                    shot * self.n_lines_epi * self.n_slice_epi,
                    (shot + 1) * self.n_lines_epi * self.n_slice_epi,
                    dtype=np.int32,
                )
                for shot in chunk
            ]
        )

        chunk_data = np.moveaxis(
            chunk_data.view(np.float32), 1, 2
        )  # put the coil axis after the readout axis
        acq_chunk = data_loader._dataset["data"][shots]
        acq_chunk["data"] = chunk_data.reshape(*acq_chunk.shape, -1)
        data_loader._dataset["data"][shots] = acq_chunk
