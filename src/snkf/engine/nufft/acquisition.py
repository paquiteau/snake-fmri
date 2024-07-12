r"""Engine That perform the acquisition of MRI data using the signal model.

.. raw ::
                      IO stuff         Apply signal model
parallel_acquire --> acquire_ksp --> acquire_ksp1  --> acquire_ksp -- > parallel_acquire
                 --> acquire_ksp --> acquire_ksp1
                 --> acquire_ksp --> acquire_ksp1


"""

import gc
import logging
import os
from collections.abc import Generator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing.managers import SharedMemoryManager

import ismrmrd as mrd
import numpy as np
from mrinufft import get_operator
from mrinufft.operators.base import FourierOperatorBase

from .._meta import MethodRegister, batched
from ..phantom import DynamicData, Phantom, PropTissueEnum
from ..simulation import SimConfig
from .parallel import ArrayProps
from .utils import get_contrast_gre

log = logging.getLogger(__name__)


acquire_register = MethodRegister("acq_nc")


def iter_traj_stacked(
    dataset: mrd.Dataset,
    sim_conf: SimConfig,
    shot_idx: Sequence[int],
    backend: str = "gpunufft",
) -> FourierOperatorBase:
    """
    Generate an updated nufft operator. Assume that the trajectory is a 2D stacked.

    Parameters
    ----------
    dataset : ismrmrd.Dataset
        The dataset object.
    shot_idx : int or Iterable[int]
        The index of the shot to be acquired.
    backend : str, optional
        The backend to use, by default "gpunufft".

    Yields
    ------
    nufft : mrinufft.FourierOperatorBase
        The operator to perform the forward model acquisition.
    """
    from mrinufft.operators.stacked import traj3d2stacked

    shape = sim_conf.shape

    if not isinstance(shot_idx, Sequence):
        shot_idx = [shot_idx]
    head = dataset._dataset["data"][0]["head"]
    n_samples = head["number_of_samples"]
    ndim = head["trajectory_dimensions"]

    smaps = None
    n_coils = sim_conf.hardware.n_coils
    try:
        smaps = dataset.read_image("smaps", 0).data
        log.info(f"Sensitivity maps {smaps.shape} found in the dataset.")
    except LookupError:
        log.warning("No sensitivity maps found in the dataset.")
        n_coils = 1

    traj = dataset._dataset["data"][0]["traj"].reshape(n_samples, ndim)
    traj2d, z_index = traj3d2stacked(traj, dim_z=shape[-1], n_samples=n_samples)

    nufft = get_operator(f"stacked-{backend}")(
        traj2d,
        shape=shape,
        z_index=z_index,
        density=False,
        smaps=smaps,
        n_coils=n_coils,
    )

    for s in shot_idx:
        traj = dataset._dataset["data"][s]["traj"].reshape(n_samples, ndim)
        traj2d, z_index = traj3d2stacked(traj, dim_z=shape[-1], n_samples=n_samples)
        nufft.operator.samples = traj2d  # update inplace, avoid recreation of operator.
        nufft.z_index = z_index

        yield nufft


def iter_traj_nufft(
    dataset: mrd.Dataset,
    sim_conf: SimConfig,
    shot_idx: Sequence[int],
    backend: str = "gpunufft",
) -> FourierOperatorBase:
    """Generate an updated nufft operator. Assumes the trajectory is a full nufft."""
    smaps = ...  # TODO extract Smaps from dataset
    shape = ...  # FIXME Extract the shape from the dataset
    # get the shape
    if not isinstance(shot_idx, Sequence):
        shot_idx = [shot_idx]
    head = dataset._dataset["data"][shot_idx[0]]["head"]
    n_samples = head["number_of_samples"]
    ndim = head["trajectory_dimensions"]

    traj = dataset._dataset["data"][shot_idx[0]]["traj"].reshape(n_samples, ndim)

    nufft = get_operator(f"stacked-{backend}")(
        traj, shape=shape, density=False, smaps=smaps
    )

    for s in shot_idx:
        traj = dataset._dataset["data"][s]["traj"].reshape(n_samples, ndim)
        nufft.operator.set_pts(traj)  # update inplace, avoid recreation of operator.
        yield nufft


@acquire_register("T2s")
def acquire_ksp(
    phantom: Phantom,
    dyn_datas: list[DynamicData],
    sim_conf: SimConfig,
    fourier_op_iterator: Generator[FourierOperatorBase],
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
def acquire_ksp(  # noqa: F811
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


def acquire_ksp(  # noqa: F811
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
