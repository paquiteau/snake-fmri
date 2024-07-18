"""Engines are responsible for the acquisition of Kspace."""

import gc
import os
from typing import Any
from collections.abc import Generator, Sequence, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing.managers import SharedMemoryManager

import ismrmrd as mrd
import numpy as np
from mrinufft.operators import FourierOperatorBase, get_operator
from numpy.typing import NDArray

from snkf._meta import LogMixin, batched
from snkf.phantom import DynamicData, Phantom, PropTissueEnum
from snkf.simulation import SimConfig

from .parallel import ArrayProps
from .utils import fft, get_contrast_gre, get_ideal_phantom


class BaseAcquisitionEngine(LogMixin):
    """Base acquisition engine.

    Specific step can be overwritten in subclasses.
    """

    def __init__(self, mode: str = "T2s", snr: float = 10.0):
        self.mode = mode
        self.snr = snr

    def _get_chunk_list(
        self, dataset: mrd.Dataset, hdr: mrd.xsd.ismrmrdHeader
    ) -> Sequence[int]:
        return range(dataset.number_of_acquisitions())

    def _job_trajectories(
        self,
        dataset: mrd.Dataset,
        hdr: mrd.xsd.ismrmrdHeader,
        sim_conf: SimConfig,
        chunk: Sequence[int],
    ) -> NDArray:
        raise NotImplementedError

    @staticmethod
    def _job_get_T2s_decay(
        dwell_time_ms: int, echo_idx: int, n_samples: int, phantom: Phantom
    ) -> NDArray:
        t = dwell_time_ms * (np.arange(n_samples, dtype=np.float32) - echo_idx)
        return np.exp(-t[None, :] / phantom.props[:, PropTissueEnum.T2s, None])

    @staticmethod
    def _job_model_T2s(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: NDArray,  # (Chunksize, N, 3)
        smaps: NDArray,
    ) -> NDArray:
        raise NotImplementedError

    @staticmethod
    def _job_model_simple(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: NDArray,  # (Chunksize, N, 3)
        smaps: NDArray,
    ) -> NDArray:
        raise NotImplementedError

    def _write_chunk_data(
        self, dataset: mrd.Dataset, chunk: Sequence[int], chunk_data: NDArray
    ) -> None:
        raise NotImplementedError

    def _acquire_ksp_job(
        self,
        filename: os.PathLike,
        sim_conf: SimConfig,
        chunk: Sequence[int],
        shared_phantom_props: tuple[ArrayProps] = None,
        mode: str = "T2s",
        **kwargs: Mapping[str, Any],
    ) -> None:
        """Entry point for worker.

        This handles the io part (Read dataset, write partial k-space),
        and dispatch to specialized functions
        for getting the k-space.

        """
        dataset = mrd.Dataset(filename)
        hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())
        # Get the Phantom, SimConfig, and all ...
        ddatas = DynamicData.all_from_mrd_dataset(dataset)
        # sim_conf = SimConfig.from_mrd_dataset(dataset)
        for d in ddatas:  # only keep the dynamic data that are in the chunk
            d.data = d.data[:, chunk]
        trajs = self._job_trajectories(dataset, hdr, sim_conf, chunk)

        _job_model = getattr(self, f"_job_model_{mode}")

        smaps = None
        try:
            smaps = dataset.read_image("smaps", 0).data
            self.log.info(
                f"Sensitivity maps {smaps.shape}, {smaps.dtype} found in the dataset."
            )
        except LookupError:
            self.log.warning("No sensitivity maps found in the dataset.")

        if shared_phantom_props is None:
            phantom = Phantom.from_mrd_dataset(dataset)
            ksp = _job_model(phantom, ddatas, sim_conf, trajs, smaps, **kwargs)
        else:
            with Phantom.from_shared_memory(*shared_phantom_props) as phantom:
                ksp = _job_model(phantom, ddatas, sim_conf, trajs, smaps, **kwargs)

        filename = os.path.join(sim_conf.tmp_dir, f"partial_{chunk[0]}.npy")
        np.save(filename, ksp)
        return filename

    def __call__(
        self,
        filename: str,
        sim_conf: SimConfig,
        worker_chunk_size: int,
        n_workers: int,
        **kwargs: Mapping[str, Any],
    ):
        """Perform the acquisition and fill the dataset."""
        dataset = mrd.Dataset(filename, create_if_needed=True)  # writeable mode
        hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())

        phantom = Phantom.from_mrd_dataset(dataset)
        shot_idxs = self._get_chunk_list(dataset, hdr)
        chunk_list = list(batched(shot_idxs, worker_chunk_size))
        ideal_phantom = get_ideal_phantom(phantom, sim_conf)
        energy = np.mean(ideal_phantom)
        del ideal_phantom

        with SharedMemoryManager() as smm, ProcessPoolExecutor(n_workers) as executor:
            phantom_props, shms = phantom.in_shared_memory(smm)
            # TODO: also put the smaps in shared memory
            futures = {
                executor.submit(
                    self._acquire_ksp_job,
                    filename,
                    sim_conf,
                    chunk,
                    shared_phantom_props=phantom_props,
                    mode=self.mode,
                    **kwargs,
                ): chunk
                for chunk in chunk_list
            }
            for future in as_completed(futures):
                chunk = futures[future]
                self.log.info(f"Done with chunk {min(chunk)}-{max(chunk)}")
                try:
                    filename = future.result()
                except Exception as exc:
                    self.log.error(f"Error in chunk {min(chunk)}-{max(chunk)}")
                    dataset.close()
                    self.log.error("Closing the dataset, raising the error.")
                    raise exc
                chunk_ksp = np.load(filename)
                # Add noise
                noise = 1j * sim_conf.rng.standard_normal(
                    size=chunk_ksp.size, dtype=np.float32
                )
                noise += sim_conf.rng.standard_normal(
                    size=chunk_ksp.size, dtype=np.float32
                )
                noise *= energy / self.snr
                chunk_ksp += noise.reshape(chunk_ksp.shape)
                self._write_chunk_data(
                    dataset,
                    chunk,
                    chunk_ksp,
                )
                os.remove(filename)
                gc.collect()
        dataset.close()


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
