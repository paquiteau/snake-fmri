"""Acquisition engine for Cartesian trajectories."""

from collections.abc import Sequence
from copy import deepcopy

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray

from snkf.phantom import DynamicData, Phantom
from snkf.simulation import SimConfig

from .base import BaseAcquisitionEngine
from .utils import fft, get_contrast_gre


class EPIAcquisitionEngine(BaseAcquisitionEngine):
    """Acquisition engine for EPI base trajectories."""

    def _get_chunk_list(
        self, dataset: mrd.Dataset, hdr: mrd.xsd.ismrmrdHeader
    ) -> Sequence[int]:
        limits = hdr.encoding[0].encodingLimits
        self.n_lines_epi = limits.kspace_encoding_step_1.maximum

        n_epi = dataset.number_of_acquisitions() // self.n_lines_epi
        return range(n_epi)

    def _job_trajectories(
        self,
        dataset: mrd.Dataset,
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
        traj = np.zeros((len(chunk), n_lines_epi, readout_length, 3), dtype=np.int32)
        for k, s in enumerate(chunk):
            for i in range(n_lines_epi):
                acq = dataset.read_acquisition(s * n_lines_epi + i)
                traj[k, i] = acq.traj.astype(np.int32, copy=False).reshape(-1, 3)

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
        echo_idx = np.argmin(
            np.sum(
                abs(
                    trajectories[0].reshape(-1, 3)
                    - (shape[0] // 2, shape[1] // 2, shape[2] // 2)
                )
                ** 2
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
            print(phantom_state.shape, phantom_state.dtype)
            if smaps is None:
                ksp = fft(phantom_state[None, ...], axis=(-3, -2, -1))
            else:
                ksp = fft(phantom_state[None, ...] * smaps, axis=(-3, -2, -1))
            flat_epi = epi_2d.reshape(-1, 3)
            for c in range(sim_conf.hardware.n_coils):
                ksp_coil = ksp[c]
                print(ksp_coil.shape, ksp.shape)
                a = ksp_coil[tuple(flat_epi.T)]
                final_ksp[i, c] = a.reshape(
                    trajectories.shape[-3],
                    trajectories.shape[-2],
                )
        return final_ksp

    def _write_chunk_data(
        self, dataset: mrd.Dataset, chunk: Sequence[int], chunk_data: NDArray
    ) -> None:
        for i, shot in enumerate(chunk):
            for ii in range(self.n_lines_epi):
                acq = dataset.read_acquisition(shot * self.n_lines_epi + ii)
                acq.data[:] = chunk_data[i, :, ii]
                dataset.write_acquisition(acq, shot * self.n_lines_epi + ii)
