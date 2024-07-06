r"""Engine That perform the acquisition of MRI data using the signal model.

.. raw ::
                      IO stuff         Apply signal model
parallel_acquire --> acquire_ksp --> acquire_ksp1  --> acquire_ksp -- > parallel_acquire
                 --> acquire_ksp --> acquire_ksp1
                 --> acquire_ksp --> acquire_ksp1


"""
from copy import deepcopy
from tqdm import tqdm
import itertools
import os
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps
from multiprocessing.managers import SharedMemoryManager
from typing import Any

import ismrmrd as mrd
import numpy as np
from numpy.typing import NDArray
from mrinufft import FourierOperatorBase, get_operator

from ..phantom import DynamicData, Phantom, PropTissueEnum
from ..simulation import SimConfig
from .parallel import ArrayProps

def get_contrast_gre(
    phantom: Phantom, FA: NDArray, TE: NDArray, TR: NDArray
) -> NDArray:
    """Compute the GRE contrast at TE."""
    return (
        np.sin(FA)
        * np.exp(-TE / phantom.tissue_properties[:,PropTissueEnum.T2])
        * (1 - np.exp(-TR / phantom.tissue_properties[:,PropTissueEnum.T1]))
        / (1 - np.cos(FA) * np.exp(-TR / phantom.tissue_properties[:,PropTissueEnum.T1]))
    )

def iter_traj_stacked(
    dataset: mrd.Dataset, shot_idx: Sequence[int], backend: str = "gpunufft"
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

    smaps = ...  # TODO extract Smaps from dataset
    shape = ...  # FIXME Extract the shape from the dataset
    # get the shape
    if not isinstance(shot_idx, Sequence):
        shot_idx = [shot_idx]
    head = dataset._dataset["data"][0]["head"]
    n_samples = head["number_of_samples"]
    ndim = head["trajectory_dimensions"]

    traj = dataset._dataset["data"][0]["traj"].reshape(n_samples, ndim)
    traj2d, z_index = traj3d2stacked(traj, dim_z=shape[-1], n_samples=n_samples)

    nufft = get_operator(f"stacked-{backend}")(
        traj2d, shape=shape, z_index=z_index, density=False, smaps=smaps
    )

    for s in shot_idx:
        traj = dataset._dataset["data"][s]["traj"].reshape(n_samples, ndim)
        traj2d, z_index = traj3d2stacked(traj, dim_z=shape[-1], n_samples=n_samples)
        nufft.operator.set_pts(traj2d)  # update inplace, avoid recreation of operator.
        nufft.z_index = z_index

        yield nufft


def iter_traj_nufft(
    dataset: mrd.Dataset, shot_idx: Sequence[int], backend: str = "gpunufft"
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


ACQUIRE_METHODS = {}


def acquire_register[T, V](t2s: bool, stacked: bool) -> Callable:
    """Register methods for the acquisition."""
    def decorator(func: Callable[T,V]) -> Callable[T,V]:
        ACQUIRE_METHODS[(t2s, stacked)] = func

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any)->V:
            return func(*args, **kwargs)

        return wrapper

    return decorator



@acquire_register(t2s=True)
def acquire_ksp1(
    phantom: Phantom,
    dyn_data: DynamicData,
    sim_conf: SimConfig,
    fourier_op_iterator: Generator[FourierOperatorBase],
    chunk_size: int,
    n_samples: int,
) -> np.ndarray:
    """Acquire k-space data."""
    final_ksp = np.zeros(chunk_size, sim_conf.n_coils, n_samples)
    # (n_tissues_true, n_samples) Filter the tissues that have NaN Values (abstract tissues.)
    t = sim_conf.hardware.dwell_time * np.arange(n_samples) - (
        n_samples // 2 if sim_conf.in_out else 0
    )
    t2s_decay = np.exp(-t / phantom.tissues_properties[PropTissueEnum.T2s])
    for i, nufft in enumerate(fourier_op_iterator):
        phantom = deepcopy(phantom)
        for dyn_data in list[DynamicData]:
            phantom = dyn_data.func(dyn_data.data, phantom, sim_conf)
        # Apply the contrast tissue-wise
        contrast = get_contrast_gre(phantom, sim_conf.seq.FA, sim_conf.seq.TE, sim_conf.seq.TR,)
        phantom_state = phantom.tissue_masks * contrast
        ksp = nufft.op(phantom_state)
        # apply the T2s and sum over tissues
        final_ksp[i] = np.einsum("kij, kj-> ij", ksp, t2s_decay)  # (n_coils, n_samples)
    return final_ksp

@acquire_register(t2s=False)
def acquire_ksp2(
    phantom: Phantom,
    dyn_data: list[DynamicData],
    sim_conf: SimConfig,
    fourier_op_iterator: Generator[FourierOperatorBase],
    chunk_size: int,
    n_samples: int,
) -> np.ndarray:
    """Acquire k-space data. No T2s decay."""
    final_ksp = np.zeros(chunk_size, sim_conf.n_coils, n_samples)
    # (n_tissues_true, n_samples) Filter the tissues that have NaN Values (abstract tissues.)
    for i, nufft in enumerate(fourier_op_iterator):
        phantom = deepcopy(phantom)
        for dyn_data in list[DynamicData]:
            phantom = dyn_data.func(dyn_data.data, phantom, sim_conf)
        # reduced the array, we dont have batch tissues !
        contrast = get_contrast_gre(phantom, sim_conf.seq.FA, sim_conf.seq.TE, sim_conf.seq.TR,)
        phantom_state = np.sum(phantom.tissue_masks * contrast)
        final_ksp[i] = nufft.op(phantom_state)
        # apply the T2s and sum over tissues.
    return final_ksp


def acquire_ksp(
    filename: os.PathLike,
    chunk: Sequence[int],
    shared_phantom_props: tuple[ArrayProps] = None,
    backend:str="gpunufft",
)->None:
    """Entry point for worker.

    This handles the io part (Read dataset, write partial k-space),
    and dispatch to specialized functions
    for getting the k-space.

    """
    dataset = mrd.Dataset(filename)
    # Get the Phantom, SimConfig, and all ...
    dyn_data = DynamicData.from_mrd_dataset(dataset, chunk)
    sim_conf = SimConfig.from_mrd_dataset(dataset)

    n_samples = dataset._dataset["data"][chunk[0]]["head"]["number_of_samples"]
    # TODO create other iterator for cartesian / 3d stacked
    fourier_op_iterator = iter_traj_stacked(dataset, chunk, backend=backend)

    if shared_phantom_props is None:
        phantom = Phantom.from_mrd_dataset(dataset)
        ksp = acquire_ksp1(
            phantom,
            dyn_data,
            sim_conf,
            fourier_op_iterator,
            len(chunk),
            n_samples,
        )
    else:
        with Phantom.from_shared_memory(shared_phantom_props) as phantom:
            ksp = acquire_ksp1(dataset, chunk, phantom)
    filename = os.path.join(sim_conf.tmp_dir, f"partial_{chunk[0]}.npy")
    np.save(filename, ksp)
    return filename

def parallel_acquire(filename:str, worker_chunk_size:int, n_workers:int)-> None:
    """ACquire the k-space data in parallel."""
    # estimat chunk size from the dataset, split the dataset in n_worker
    # Run the acquire_ksp function in parallel, collect the resulting files and merge them.

    dataset = mrd.Dataset(filename, create_if_needed=False)
    n_shots = dataset.number_of_acquisitions()

    phantom = Phantom.from_mrd_dataset(dataset)
    phantom_props = phantom.to_shared_memory()

    chunk_list = list(itertools.batched(range(n_shots), worker_chunk_size))
    #
    with (SharedMemoryManager(), ProcessPoolExecutor(n_workers)) as (smm, executor):
        futures = {
            executor.submit(acquire_ksp, filename, chunk, phantom_props): chunk
            for chunk in chunk_list
        }
        for future in as_completed(futures):
            chunk = futures[future]
            chunk_ksp = np.load(future.result())
            for i, shot in enumerate(chunk):
                dataset._dataset["data"][shot]["data"] = chunk_ksp[i]
    dataset.close()
   
