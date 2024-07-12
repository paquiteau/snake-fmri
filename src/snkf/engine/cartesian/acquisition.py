"""Engine for cartesian acquisition of data."""

import gc
import logging
import os
from collections.abc import Generator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import ismrmrd as mrd
import scipy as sp

from .._meta import MethodRegister
from ..phantom import Phantom, PropTissueEnum, DynamicData
from ..simulation import SimConfig
from ..utils import batched

log = logging.getLogger(__name__)


acquire_register = MethodRegister("acquire_cartesian")


def iter_traj(
        dataset: mrd.Dataset,
        sim_conf: SimConfig,
        shot: Sequence[int],
) ->FFT_SENSE:

@acquire_register("T2s")
def acquire_ksp(
    phantom: Phantom,
    dyn_datas: list[DynamicData],
    sim_conf: SimConfig,
    fourier_op_iterator: Generator,
    chunk_size: int,
    n_samples: int,
    center_sample_idx: int,
) -> np.ndarray:
    """Acquire k-space data."""
    final_ksp = np.zeros(
        (chunk_size, sim_conf.hardware.n_coils, n_samples), dtype=np.complex64
    )
    # (n_tissues_true, n_samples) Filter the tissues that have NaN Values.
    t = sim_conf.hardware.dwell_time_ms * (
        np.arange(n_samples, dtype=np.float32) - center_sample_idx
    )
    t2s_decay = np.exp(
        -t[None, :] / phantom.tissue_properties[:, PropTissueEnum.T2s, None]
    )
    for i, nufft in enumerate(fourier_op_iterator):
        phantom = deepcopy(phantom)
        # for dyn_data in dyn_datas:
        #     phantom = dyn_data.apply(phantom, sim_conf)
        # Apply the contrast tissue-wise
        contrast = get_contrast_gre(
            phantom,
            sim_conf.seq.FA,
            sim_conf.seq.TE,
            sim_conf.seq.TR,
        )
        phantom_state = (
            1000
            * contrast[(..., *([None] * len(phantom.anat_shape)))]
            * phantom.tissue_masks
        )
        nufft.n_batchs = len(phantom_state)
        ksp = nufft.op(phantom_state)
        # apply the T2s and sum over tissues
        # final_ksp[i] = np.sum(ksp * t2s_decay[:, None, :], axis=0)
        final_ksp[i] = np.einsum("kij, kj-> ij", ksp, t2s_decay)
    return final_ksp


@acquire_register("simple")
def acquire_ksp(
    phantom: Phantom,
    dyn_data: list[DynamicData],
    sim_conf: SimConfig,
    fourier_op_iterator: Generator[FourierOperatorBase],
    chunk_size: int,
    n_samples: int,
    center_sample: int,
) -> np.ndarray:
    """Acquire k-space data. No T2s decay."""
    final_ksp = np.zeros(
        (chunk_size, sim_conf.hardware.n_coils, n_samples), dtype=np.complex64
    )
    # (n_tissues_true, n_samples) Filter the tissues that have NaN Values
    for i, nufft in enumerate(fourier_op_iterator):
        phantom = deepcopy(phantom)
        # for dyn_data in list[DynamicData]:
        #     phantom = dyn_data.func(dyn_data.data, phantom, sim_conf)
        # reduced the array, we dont have batch tissues !
        contrast = get_contrast_gre(
            phantom,
            sim_conf.seq.FA,
            sim_conf.seq.TE,
            sim_conf.seq.TR,
        )
        phantom_state = np.sum(
            1000
            * contrast[(..., *([None] * len(phantom.anat_shape)))]
            * phantom.tissue_masks,
            axis=0,
        )
        final_ksp[i] = nufft.op(phantom_state)
    return final_ksp


def acquire_ksp(
    filename: os.PathLike,
    sim_conf: SimConfig,
    chunk: Sequence[int],
    shared_phantom_props: tuple[ArrayProps] = None,
    backend: str = "gpunufft",
    mode: str = "T2s",
) -> None:
    """Entry point for worker.

    This handles the io part (Read dataset, write partial k-space),
    and dispatch to specialized functions
    for getting the k-space.

    """
    dataset = mrd.Dataset(filename)
    # Get the Phantom, SimConfig, and all ...
    dyn_data = DynamicData.from_mrd_dataset(dataset, chunk)
    # sim_conf = SimConfig.from_mrd_dataset(dataset)

    n_samples = dataset._dataset["data"][chunk[0]]["head"]["number_of_samples"]
    center_sample = dataset._dataset["data"][chunk[0]]["head"]["center_sample"]
    log.info("N_samples %s, center: %s", n_samples, center_sample)
    # TODO create other iterator for cartesian / 3d stacked

    fourier_op_iterator = iter_traj_stacked(dataset, sim_conf, chunk, backend=backend)

    _acquire = acquire_register.registry["acq_nc"][mode]

    if shared_phantom_props is None:
        phantom = Phantom.from_mrd_dataset(dataset)
        ksp = _acquire(
            phantom,
            dyn_data,
            sim_conf,
            fourier_op_iterator,
            len(chunk),
            n_samples,
            center_sample,
        )
    else:
        with Phantom.from_shared_memory(*shared_phantom_props) as phantom:
            # FIXME do the dispatch for acquire_ksp1 or acquire_ksp2
            ksp = _acquire(
                phantom,
                dyn_data,
                sim_conf,
                fourier_op_iterator,
                len(chunk),
                n_samples,
                center_sample,
            )

    filename = os.path.join(sim_conf.tmp_dir, f"partial_{chunk[0]}.npy")
    np.save(filename, ksp)
    return filename


def parallel_acquire(
    filename: str,
    sim_conf: SimConfig,
    worker_chunk_size: int,
    n_workers: int,
    mode: str = "T2s",
) -> None:
    """Acquire the k-space data in parallel.

    1. An existing file (with empty acquisition data) is split in chunk
    2. Each chunk is processed using a concurrent.futures interfaces
    3. When a chunk result is available (npy file) it is written back to original file.

    """
    # TODO Recreate SimConfig object from dataset ?
    dataset = mrd.Dataset(filename, create_if_needed=True)  # writeable mode
    n_shots = dataset.number_of_acquisitions()
    log.info("Acquiring %d shots", n_shots)

    phantom = Phantom.from_mrd_dataset(dataset)

    chunk_list = list(batched(range(n_shots), worker_chunk_size))
    #
    with SharedMemoryManager() as smm, ProcessPoolExecutor(n_workers) as executor:
        phantom_props, shms = phantom.in_shared_memory(smm)
        # TODO: also put the smaps in shared memory
        futures = {
            executor.submit(
                acquire_ksp,
                filename,
                sim_conf,
                chunk,
                shared_phantom_props=phantom_props,
                mode=mode,
            ): chunk
            for chunk in chunk_list
        }
        for future in as_completed(futures):
            chunk = futures[future]
            log.info(f"Done with chunk {min(chunk)}-{max(chunk)}")
            filename = future.result()
            chunk_ksp = np.load(filename)
            log.info(f"{np.max(abs(chunk_ksp))}")
            for i, shot in enumerate(chunk):
                acq = dataset.read_acquisition(shot)
                acq.data[:] = chunk_ksp[i]
                dataset.write_acquisition(acq, shot)
            del chunk_ksp
            os.remove(filename)
            gc.collect()
    dataset.close()
