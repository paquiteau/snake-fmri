"""Engine for cartesian acquisition of data."""

import gc
import logging
import os
from collections.abc import Generator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing.managers import SharedMemoryManager

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray
from snkf._meta import MethodRegister, batched
from snkf.phantom import DynamicData, Phantom, PropTissueEnum
from snkf.simulation import SimConfig
from snkf.mrd_utils import ACQ

from ..parallel import ArrayProps
from ..utils import get_contrast_gre

from .operators import fft

log = logging.getLogger(__name__)


acquire_register = MethodRegister("acq_cartesian")


def extract_trajectory(
    dataset: mrd.Dataset,
    sim_conf: SimConfig,
    epi_idx: Sequence[int],
    n_lines_in_epi: int,
    readout_length: int,
) -> np.ndarray:
    """Generate the fourier operator by iterating the dataset."""
    if not isinstance(epi_idx, Sequence):
        shot = [epi_idx]

    traj = np.zeros((len(epi_idx), n_lines_in_epi, readout_length, 3), dtype=np.int32)
    for k, s in enumerate(epi_idx):
        for i in range(n_lines_in_epi):
            acq = dataset.read_acquisition(s * n_lines_in_epi + i)
            traj[k, i] = acq.traj.astype(np.int32, copy=False).reshape(-1, 3)

    return traj


@acquire_register
def T2s(
    phantom: Phantom,
    dyn_datas: list[DynamicData],
    sim_conf: SimConfig,
    trajectories: NDArray,  # (Chunksize, N, 3)
    smaps: NDArray,
) -> np.ndarray:
    """Acquire k-space data. With T2s decay."""
    readout_length = trajectories.shape[-2]
    n_lines_epi = trajectories.shape[-3]
    log.info(f"Acquiring {len(trajectories)} k-space data with T2S model.")

    final_ksp = np.zeros(
        (
            len(trajectories),
            sim_conf.hardware.n_coils,
            n_lines_epi,
            readout_length,
        ),
        dtype=np.complex64,
    )

    n_samples = readout_length * n_lines_epi
    t = sim_conf.hardware.dwell_time_ms * (
        np.arange(n_samples, dtype=np.float32) - n_samples // 2
    )

    t2s_decay = np.exp(
        -t[None, :] / phantom.tissue_properties[:, PropTissueEnum.T2s, None]
    )

    for i, epi_2d in enumerate(trajectories):
        frame_phantom = deepcopy(phantom)
        for dyn_data in dyn_datas:
            frame_phantom = dyn_data.func(frame_phantom, dyn_data.data, i)

        contrast = get_contrast_gre(
            phantom,
            sim_conf.seq.FA,
            sim_conf.seq.TE,
            sim_conf.seq.TR,
        )
        phantom_state = (
            contrast[(..., *([None] * len(phantom.anat_shape)))] * phantom.tissue_masks
        )

        ksp = fft(phantom_state[:, None, ...] * smaps, axis=(-3, -2, -1))
        flat_epi = epi_2d.reshape(-1, 3)
        for c in range(sim_conf.hardware.n_coils):
            ksp_coil_sum = np.zeros((n_lines_epi * readout_length), dtype=np.complex64)
            for b in range(phantom_state.shape[0]):
                ksp_coil_sum += ksp[b, c][tuple(flat_epi.T)] * t2s_decay[b]
            final_ksp[i, c] = ksp_coil_sum.reshape((n_lines_epi, readout_length))

    return final_ksp


@acquire_register
def simple(
    phantom: Phantom,
    dyn_datas: list[DynamicData],
    sim_conf: SimConfig,
    trajectories: NDArray,  # (Chunksize, N, 3)
    smaps: NDArray,
) -> np.ndarray:
    """Acquire k-space data. No T2s decay."""
    log.info(f"Acquiring {len(trajectories)} k-space data with SIMPLE model.")
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
            phantom,
            sim_conf.seq.FA,
            sim_conf.seq.TE,
            sim_conf.seq.TR,
        )
        phantom_state = np.sum(
            contrast[(..., *([None] * len(phantom.anat_shape)))] * phantom.tissue_masks,
            axis=0,
        )

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


def acquire_ksp_job(
    filename: os.PathLike,
    sim_conf: SimConfig,
    chunk: Sequence[int],
    n_lines_in_epi: int,
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
    dyn_datas = DynamicData.all_from_mrd_dataset(dataset)
    # sim_conf = SimConfig.from_mrd_dataset(dataset)
    readout_length = sim_conf.shape[2]
    fourier_op_iterator = extract_trajectory(
        dataset,
        sim_conf,
        chunk,
        n_lines_in_epi,
        readout_length,
    )

    _acquire = acquire_register.registry["acq_cartesian"][mode]

    smaps = None
    try:
        smaps = dataset.read_image("smaps", 0).data
        log.info(f"Sensitivity maps {smaps.shape}, {smaps.dtype} found in the dataset.")
    except LookupError:
        log.warning("No sensitivity maps found in the dataset.")

    if shared_phantom_props is None:
        phantom = Phantom.from_mrd_dataset(dataset)
        ksp = _acquire(phantom, dyn_datas, sim_conf, fourier_op_iterator, smaps)
    else:
        with Phantom.from_shared_memory(*shared_phantom_props) as phantom:
            ksp = _acquire(phantom, dyn_datas, sim_conf, fourier_op_iterator, smaps)

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
    hdr = mrd.xsd.CreateFromDocument(dataset.read_xml_header())

    n_lines_in_epi = hdr.encoding[0].encodingLimits.kspace_encoding_step_1.maximum

    n_epi = dataset.number_of_acquisitions() // n_lines_in_epi
    log.info("Acquiring %d epi", n_epi)

    phantom = Phantom.from_mrd_dataset(dataset)

    chunk_list = list(batched(range(n_epi), worker_chunk_size))
    #
    with SharedMemoryManager() as smm, ProcessPoolExecutor(n_workers) as executor:
        phantom_props, shms = phantom.in_shared_memory(smm)
        # TODO: also put the smaps in shared memory
        futures = {
            executor.submit(
                acquire_ksp_job,
                filename,
                sim_conf,
                chunk,
                n_lines_in_epi,
                shared_phantom_props=phantom_props,
                mode=mode,
            ): chunk
            for chunk in chunk_list
        }
        for future in as_completed(futures):
            chunk = futures[future]
            log.info(f"Done with chunk {min(chunk)}-{max(chunk)}")
            try:
                filename = future.result()
            except Exception as exc:
                log.error(f"Error in chunk {min(chunk)}-{max(chunk)}")
                dataset.close()
                log.error("Closing the dataset, raising the error.")
                raise exc
            chunk_ksp = np.load(filename)
            # FIXME
            # put each line of kspace in the right place
            for i, shot in enumerate(chunk):
                for ii in range(n_lines_in_epi):
                    acq = dataset.read_acquisition(shot * n_lines_in_epi + ii)
                    acq.data[:] = chunk_ksp[i, :, ii]
                    dataset.write_acquisition(acq, shot * n_lines_in_epi + ii)
            del chunk_ksp
            os.remove(filename)
            gc.collect()
    dataset.close()
