"""Engines are responsible for the acquisition of Kspace."""

from __future__ import annotations
import atexit
import gc
import logging
import os
from pathlib import Path
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing.managers import SharedMemoryManager
from typing import Any, ClassVar

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .._meta import MetaDCRegister, batched, EnvConfig
from ..mrd_utils import MRDLoader
from ..parallel import ArrayProps
from ..phantom import DynamicData, Phantom, PropTissueEnum
from ..simulation import SimConfig
from .utils import get_ideal_phantom, get_noise


class MetaEngine(MetaDCRegister):
    """MetaClass for engines."""

    dunder_name = "engine"


class BaseAcquisitionEngine(metaclass=MetaEngine):
    """Base acquisition engine.

    Specific step can be overwritten in subclasses.
    """

    __engine_name__: ClassVar[str]
    __registry__: ClassVar[dict[str, type[BaseAcquisitionEngine]]]
    log: ClassVar[logging.Logger]

    model: str = "simple"
    snr: float = np.inf

    def _get_chunk_list(
        self,
        data_loader: MRDLoader,
    ) -> Sequence[int]:
        return range(data_loader.n_acquisition)

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
        filename: os.PathLike,
        chunk: Sequence[int],
        shared_phantom_props: (
            tuple[str, ArrayProps, ArrayProps, ArrayProps] | None
        ) = None,
        model: str = "T2s",
        **kwargs: Mapping[str, Any],
    ) -> str:
        """Entry point for worker.

        This handles the io part (Read dataset, write partial k-space),
        and dispatch to specialized functions
        for getting the k-space.

        """
        # https://github.com/h5py/h5py/issues/712#issuecomment-562980532
        # We know that we are going to read the dataset in read-only mode in
        # this function and use the main process to write the data.
        # This is an alternative to using swmr mode, that I could not get to work.
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        with MRDLoader(filename) as data_loader:
            hdr = data_loader.header
            # Get the Phantom, SimConfig, and all ...
            sim_conf = data_loader.get_sim_conf()
            ddatas = data_loader.get_all_dynamic()
            # sim_conf = SimConfig.from_mrd_dataset(dataset)
            for d in ddatas:  # only keep the dynamic data that are in the chunk
                d.data = d.data[:, chunk]
            trajs = self._job_trajectories(data_loader, hdr, sim_conf, chunk)

            _job_model = getattr(self, f"_job_model_{model}")
            smaps = None
            if sim_conf.hardware.n_coils > 1:
                smaps = data_loader.get_smaps()
            if shared_phantom_props is None:
                phantom = data_loader.get_phantom()
                ksp = _job_model(phantom, ddatas, sim_conf, trajs, smaps, **kwargs)
            else:
                with Phantom.from_shared_memory(*shared_phantom_props) as phantom:
                    ksp = _job_model(phantom, ddatas, sim_conf, trajs, smaps, **kwargs)

        chunk_file = os.path.join(EnvConfig["SNAKE_TMP_DIR"], f"partial_{chunk[0]}.npy")
        np.save(chunk_file, ksp)
        return chunk_file

    def __call__(
        self,
        filename: os.PathLike,
        worker_chunk_size: int,
        n_workers: int,
        **kwargs: Mapping[str, Any],
    ):
        """Perform the acquisition and fill the dataset."""
        with MRDLoader(filename) as data_loader:
            sim_conf = data_loader.get_sim_conf()

            phantom = data_loader.get_phantom()
            shot_idxs = self._get_chunk_list(data_loader)

            chunk_list = list(batched(shot_idxs, worker_chunk_size))
            ideal_phantom = get_ideal_phantom(phantom, sim_conf)
            energy = np.mean(ideal_phantom)

            coil_cov = data_loader.get_coil_cov() or np.eye(sim_conf.hardware.n_coils)

            if self.snr > 0:
                coil_cov = coil_cov * energy / self.snr
        del ideal_phantom

        # https://github.com/h5py/h5py/issues/712#issuecomment-562980532
        # We know that we are going to read the dataset in read-only mode
        # and use the main process (here) to write the data.
        # This is an alternative to using swmr mode, that I could not get to work.

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        atexit.register(del_future_files)

        with (
            SharedMemoryManager() as smm,
            ProcessPoolExecutor(
                n_workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor,
            tqdm(total=len(shot_idxs)) as pbar,
            MRDLoader(filename, writeable=True) as data_loader,
        ):
            # data_loader._file.swmr_mode = True

            with open(os.path.join(EnvConfig["SNAKE_TMP_DIR"], "chunks"), "w") as f:
                f.write(",".join([str(c[0]) for c in chunk_list]))
            phantom_props, shms = phantom.in_shared_memory(smm)
            # TODO: also put the smaps in shared memory
            futures = {
                executor.submit(
                    self._acquire_ksp_job,
                    filename,
                    chunk,
                    shared_phantom_props=phantom_props,
                    model=self.model,
                    **kwargs,
                ): chunk
                for chunk in chunk_list
            }
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    f_chunk = str(future.result())
                except Exception as exc:
                    self.log.error(f"Error in chunk {min(chunk)}-{max(chunk)}")
                    raise exc
                else:
                    pbar.update(worker_chunk_size)
                chunk_ksp = np.load(f_chunk)
                # Add noise
                if self.snr > 0:
                    noise = get_noise(chunk_ksp, coil_cov, sim_conf.rng)
                    chunk_ksp += noise

                self._write_chunk_data(
                    data_loader,
                    chunk,
                    chunk_ksp,
                )
                os.remove(f_chunk)
                gc.collect()
        os.remove(os.path.join(EnvConfig["SNAKE_TMP_DIR"], "chunks"))


def del_future_files():
    """Delete the files created by the engine."""
    SNAKE_TMP_DIR = Path(os.environ.get("SNAKE_TMP_DIR", "/tmp"))
    with open(SNAKE_TMP_DIR / "chunks") as f:
        chunks = f.read().split(",")
    for chunk in chunks:
        os.remove(SNAKE_TMP_DIR / f"partial_{chunk}.npy")
    os.remove(SNAKE_TMP_DIR / "chunks")
