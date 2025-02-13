"""Engines are responsible for the acquisition of Kspace."""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from typing_extensions import dataclass_transform

from snake._meta import EnvConfig, MetaDCRegister, batched

from ...mrd_utils import MRDLoader, make_base_mrd
from ..handlers import AbstractHandler, HandlerList
from ..parallel import ArrayProps
from ..phantom import DynamicData, Phantom, PropTissueEnum
from ..sampling import BaseSampler
from ..simulation import SimConfig
from .utils import get_ideal_phantom, get_noise

AnyPath = str | Path


@dataclass_transform(kw_only_default=True)
class MetaEngine(MetaDCRegister):
    """MetaClass for engines."""

    dunder_name: ClassVar[str] = "engine"


class BaseAcquisitionEngine(metaclass=MetaEngine):
    """Base acquisition engine.

    Specific step can be overwritten in subclasses.

    Parameters
    ----------
    model : str, optional
        The model to use, by default "simple".
    snr : float, optional
        The signal to noise ratio, by default np.inf.
    slice_2d : bool, optional
        Whether to slice the 2D data, by default False.
    """

    __engine_name__: ClassVar[str]
    __registry__: ClassVar[dict[str, type[BaseAcquisitionEngine]]]
    log: ClassVar[logging.Logger]

    model: str = "simple"
    snr: float = np.inf
    slice_2d: bool = False

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
        filename: AnyPath,
        chunk: Sequence[int],
        tmp_dir: str,
        shared_phantom_props: (
            tuple[str, ArrayProps, ArrayProps, ArrayProps, ArrayProps | None] | None
        ) = None,
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
        with MRDLoader(filename, swmr=True) as data_loader:
            hdr = data_loader.header
            # Get the Phantom, SimConfig, and all ...
            sim_conf = data_loader.get_sim_conf()
            ddatas = data_loader.get_all_dynamic()
            # sim_conf = SimConfig.from_mrd_dataset(dataset)
            for d in ddatas:  # only keep the dynamic data that are in the chunk
                d.data = d.data[:, chunk]
            trajs = self._job_trajectories(data_loader, hdr, sim_conf, chunk)

            _job_model = getattr(self, f"_job_model_{self.model}")
            if shared_phantom_props is None:
                phantom = data_loader.get_phantom()
                ksp = _job_model(phantom, ddatas, sim_conf, trajs, **kwargs)
            else:
                with Phantom.from_shared_memory(*shared_phantom_props) as phantom:
                    ksp = _job_model(phantom, ddatas, sim_conf, trajs, **kwargs)

        chunk_file = os.path.join(tmp_dir, f"partial_{chunk[0]}-{chunk[-1]}.npy")
        np.save(chunk_file, ksp)
        return chunk_file

    def __call__(
        self,
        filename: AnyPath,
        sampler: BaseSampler,
        phantom: Phantom,
        sim_conf: SimConfig,
        handlers: list[AbstractHandler] | HandlerList | None = None,
        coil_cov: NDArray | None = None,
        worker_chunk_size: int = 0,
        n_workers: int = 0,
        **kwargs: Any,
    ):
        """Perform the acquisition and fill the dataset.

        Parameters
        ----------
        filename : AnyPath
            The path to the MRD file.
        sampler : BaseSampler
            The sampler to use.
        phantom : Phantom
            The phantom to use.
        sim_conf : SimConfig
            The simulation configuration.
        handlers : list[AbstractHandler] | HandlerList | None, optional
            The handlers to use, by default None.
        coil_cov : NDArray | None, optional
            The coil covariance matrix, by default None.
        worker_chunk_size : int, optional
            The size of the chunk to process for each worker, by default 0 (auto).
            Each worker w
        n_workers : int, optional
            The number of workers to use, by default 0 (auto). Half of CPU count will
            be used (This usually corresponds to the number of physical cores on the
            machine).
        kwargs : Any
            Additional keyword arguments, passed down to internal implementation.


        Notes
        -----
        This function is the main entry point for the acquisition engine.
        It will create the base dataset, and then dispatch the work to the workers.

        Specific modeling steps are implemented in the `_job_model_T2s` and
        `_job_model_simple` methods.
        """
        # Create the base dataset
        make_base_mrd(
            filename,
            sampler,
            phantom,
            sim_conf,
            handlers,
            coil_cov,
            self.model,
            self.slice_2d,
        )

        # Guesstimate the workload
        if worker_chunk_size <= 0:
            # get the number of shot
            worker_chunk_size = sampler.get_next_frame(sim_conf).shape[0]
        if n_workers <= 0:
            n_workers = mp.cpu_count() // 2

        with MRDLoader(filename) as data_loader:
            sim_conf = data_loader.get_sim_conf()

            phantom = data_loader.get_phantom()
            shot_idxs = self._get_chunk_list(data_loader)

            chunk_list = list(batched(shot_idxs, worker_chunk_size))
            ideal_phantom = get_ideal_phantom(phantom, sim_conf)

            coil_cov = data_loader.get_coil_cov() or np.eye(sim_conf.hardware.n_coils)
            if self.snr > 0:
                energy = np.mean(ideal_phantom**2)
                coil_cov = coil_cov * energy / self.snr

        del ideal_phantom

        # https://github.com/h5py/h5py/issues/712#issuecomment-562980532
        # We know that we are going to read the dataset in read-only mode
        # and use the main process (here) to write the data.
        # This is an alternative to using swmr mode, that I could not get to work.

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        if n_workers > 1:
            Executor = ProcessPoolExecutor(
                n_workers, mp_context=mp.get_context(self.__mp_mode__)
            )
        else:
            Executor = ThreadPoolExecutor(max_workers=1)
        with (
            SharedMemoryManager() as smm,
            Executor as executor,
            tqdm(total=len(shot_idxs)) as pbar,
            MRDLoader(filename, writeable=True) as data_loader,
            TemporaryDirectory(
                dir=EnvConfig["SNAKE_TMP_DIR"], prefix="snake-"
            ) as tmp_chunk_dir,
        ):
            # data_loader._file.swmr_mode = True
            phantom_props, shms = phantom.in_shared_memory(smm)
            # TODO: also put the smaps in shared memory
            futures = {
                executor.submit(
                    self._acquire_ksp_job,
                    filename,
                    chunk_id,
                    tmp_dir=tmp_chunk_dir,
                    shared_phantom_props=phantom_props,
                    slice_2d=self.slice_2d,
                    **kwargs,
                ): chunk_id
                for chunk_id in chunk_list
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
