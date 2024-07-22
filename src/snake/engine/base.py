"""Engines are responsible for the acquisition of Kspace."""

from __future__ import annotations
import gc
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.managers import SharedMemoryManager
from typing import Any
import logging
import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray
from typing import ClassVar, Literal
from .._meta import batched, MetaDCRegister
from ..mrd_utils import parse_sim_conf, read_mrd_header, load_coil_cov, load_smaps
from ..phantom import DynamicData, Phantom, PropTissueEnum
from ..simulation import SimConfig

from ..parallel import ArrayProps
from .utils import get_ideal_phantom, get_noise


class MetaEngine(MetaDCRegister):
    """MetaClass for engines."""

    dunder_name = "engine"


class BaseAcquisitionEngine(metaclass=MetaEngine):
    """Base acquisition engine.

    Specific step can be overwritten in subclasses.
    """

    __engine_name__: ClassVar[str]
    __registry__: ClassVar[dict[str, BaseAcquisitionEngine]]
    log: logging.Logger
    mode: str = "simple"
    snr: float = np.inf

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
        dwell_time_ms: float,
        echo_idx: int,
        n_samples: int,
        phantom: Phantom,
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
        *args: Any,
        **kwargs: Any,
    ) -> NDArray:
        raise NotImplementedError

    @staticmethod
    def _job_model_simple(
        phantom: Phantom,
        dyn_datas: list[DynamicData],
        sim_conf: SimConfig,
        trajectories: NDArray,  # (Chunksize, N, 3)
        smaps: NDArray,
        *args: Any,
        **kwargs: Any,
    ) -> NDArray:
        raise NotImplementedError

    def _write_chunk_data(
        self, dataset: mrd.Dataset, chunk: Sequence[int], chunk_data: NDArray
    ) -> None:
        raise NotImplementedError

    def _acquire_ksp_job(
        self,
        filename: os.PathLike | str,
        chunk: Sequence[int],
        shared_phantom_props: (
            tuple[str, ArrayProps, ArrayProps, ArrayProps] | None
        ) = None,
        mode: str = "T2s",
        **kwargs: Mapping[str, Any],
    ) -> str:
        """Entry point for worker.

        This handles the io part (Read dataset, write partial k-space),
        and dispatch to specialized functions
        for getting the k-space.

        """
        dataset = mrd.Dataset(filename)
        hdr = read_mrd_header(dataset)
        # Get the Phantom, SimConfig, and all ...
        sim_conf = parse_sim_conf(hdr)
        ddatas = DynamicData.all_from_mrd_dataset(dataset)
        # sim_conf = SimConfig.from_mrd_dataset(dataset)
        for d in ddatas:  # only keep the dynamic data that are in the chunk
            d.data = d.data[:, chunk]
        trajs = self._job_trajectories(dataset, hdr, sim_conf, chunk)

        _job_model = getattr(self, f"_job_model_{mode}")

        smaps = load_smaps(dataset)
        if shared_phantom_props is None:
            phantom = Phantom.from_mrd_dataset(dataset)
            ksp = _job_model(phantom, ddatas, sim_conf, trajs, smaps, **kwargs)
        else:
            with Phantom.from_shared_memory(*shared_phantom_props) as phantom:
                ksp = _job_model(phantom, ddatas, sim_conf, trajs, smaps, **kwargs)

        chunk_file = os.path.join(sim_conf.tmp_dir, f"partial_{chunk[0]}.npy")
        np.save(chunk_file, ksp)
        return chunk_file

    def __call__(
        self,
        filename: str,
        worker_chunk_size: int,
        n_workers: int,
        **kwargs: Mapping[str, Any],
    ):
        """Perform the acquisition and fill the dataset."""
        dataset = mrd.Dataset(filename, create_if_needed=True)  # writeable mode
        hdr = read_mrd_header(dataset)
        sim_conf = parse_sim_conf(hdr)

        phantom = Phantom.from_mrd_dataset(dataset)
        shot_idxs = self._get_chunk_list(dataset, hdr)
        chunk_list = list(batched(shot_idxs, worker_chunk_size))
        ideal_phantom = get_ideal_phantom(phantom, sim_conf)
        energy = np.mean(ideal_phantom)

        coil_cov = load_coil_cov(dataset) or np.eye(sim_conf.hardware.n_coils)

        coil_cov = coil_cov * energy / self.snr

        del ideal_phantom

        with SharedMemoryManager() as smm, ProcessPoolExecutor(n_workers) as executor:
            phantom_props, shms = phantom.in_shared_memory(smm)
            # TODO: also put the smaps in shared memory
            futures = {
                executor.submit(
                    self._acquire_ksp_job,
                    filename,
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
                    f_chunk = str(future.result())
                except Exception as exc:
                    self.log.error(f"Error in chunk {min(chunk)}-{max(chunk)}")
                    dataset.close()
                    self.log.error("Closing the dataset, raising the error.")
                    raise exc
                chunk_ksp = np.load(f_chunk)
                # Add noise
                if self.snr != np.inf:
                    noise = get_noise(chunk_ksp, coil_cov, sim_conf.rng)
                    chunk_ksp += noise
                self._write_chunk_data(
                    dataset,
                    chunk,
                    chunk_ksp,
                )
                os.remove(f_chunk)
                gc.collect()
        dataset.close()

    @classmethod
    def auto(
        cls,
        filename: os.PathLike,
        n_workers: int,
        worker_chunk_size: int,
        *args: Any,
        **kwargs: Any,
    ):
        """Get the engine specified in the file and use it."""
        hdr = read_mrd_header(filename)
        _, version, engine = hdr.acquisitionSystemInformation.systemModel.split("-")

        klass = AbstractEngine.__registry__[engine]
        return klass(*args, **kwargs)()
