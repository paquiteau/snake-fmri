"""Engines are responsible for the acquisition of Kspace."""

import gc
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.managers import SharedMemoryManager
from typing import Any

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray

from snkf._meta import LogMixin, batched
from snkf.phantom import DynamicData, Phantom, PropTissueEnum
from snkf.simulation import SimConfig

from .parallel import ArrayProps
from .utils import get_ideal_phantom


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
