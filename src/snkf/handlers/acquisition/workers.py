"""Multiprocessing module for the acquisition of data."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Generator, Mapping
from numpy.typing import NDArray

import numpy as np
from fmri.operators.fourier import FFT_Sense
from joblib.externals.loky import get_reusable_executor
from joblib import Parallel, delayed
from mrinufft import get_operator

try:
    from mrinufft.operators.interfaces.gpunufft import make_pinned_smaps
except ImportError:
    make_pinned_smaps = None

from tqdm.auto import tqdm

from snkf.simulation import SimData, LazySimArray
from snkf.base import DuplicateFilter
from ._tools import TrajectoryGeneratorType

# from mrinufft import get_operator

logger = logging.getLogger("simulation." + __name__)


def kspace_bulk_shot(
    traj_generator: Generator[np.ndarray, None, None],
    n_sim_frame: int,
    n_batch: int = 3,
) -> Generator[tuple[int, np.ndarray, list[tuple[int, int]]], None, None]:
    """Generate a stream of shot, delivered in batch.

    This function is a generator that yield a tuple of (sim_frame shots, shot_pos) where
    shot_pos is a tuple describing the absolute position

    Parameters
    ----------
    traj_factory: Callable
        A function that create a ndarray representing a kspace trajectory
    n_batch: int
        The number of shots delivered together. Typically n_batch*shot_time = sim_time
        (time for a single sim_frame)
    factory_kwargs: dict
        Parameter for the trajectory

    Yields
    ------
    tuple[int, np.ndarray, list[tuple[int, int]]
        A tuple of (sim_frame shots, shot_pos) where shot_pos is the absolute position
        (kspace_frame, shot_idx) index of the kspace frame in the full trajectory.
    """
    shots = next(traj_generator)
    shot_idx = 0
    kframe = 0
    for sim_frame_idx in range(n_sim_frame):
        if shot_idx + n_batch <= len(shots):
            yield (
                sim_frame_idx,
                shots[shot_idx : shot_idx + n_batch],
                [(kframe, i) for i in range(shot_idx, shot_idx + n_batch)],
            )
            shot_idx += n_batch
        elif shot_idx <= len(shots):
            # The last batch is incomplete, so we start a new trajectory
            # to complete it.
            new_shots = next(traj_generator)
            new_shot_idx = shot_idx + n_batch - len(shots)
            yield (
                sim_frame_idx,
                np.vstack([shots[shot_idx:], new_shots[:new_shot_idx]]),
                [(kframe, i) for i in range(shot_idx, len(shots))]
                + [(kframe + 1, i) for i in range(new_shot_idx)],
            )
            shots = new_shots
            shot_idx = new_shot_idx
            kframe += 1
        else:
            raise RuntimeError("invalid shot_idx value")


def _get_slicer(shot: np.ndarray) -> tuple[slice, ...]:
    """Return a slicer for the mask.

    Fully sampled axis are marked with a -1.
    """
    slicer = [slice(None, None, None)] * shot.shape[-1]
    accel_axis = [i for i, v in enumerate(shot[0]) if v != -1][0]
    slicer[accel_axis] = shot[0][accel_axis]
    return tuple(slicer)


def acq_cartesian(
    sim: SimData,
    trajectory_gen: TrajectoryGeneratorType,
    n_shot_sim_frame: int,
    n_kspace_frame: int,
    n_jobs: int = -1,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Acquire with cartesian stuff."""
    if sim.data_acq is None:
        raise ValueError("data acq is not available")
    scheduler = kspace_bulk_shot(trajectory_gen, sim.n_frames, n_shot_sim_frame)

    shot_batches, shot_pos, kdata_t = zip(
        *Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_cart_worker)(
                sim_frame, shot_batch, shot_pos, smaps=sim.smaps, n_coils=sim.n_coils
            )
            for sim_frame, shot_batch, shot_pos in tqdm(
                work_generator(sim.data_acq, scheduler)
            )
        )
    )

    get_reusable_executor().shutdown(wait=True)

    # reorganize the data
    kdata = np.zeros((n_kspace_frame, sim.n_coils, *sim.shape), np.complex64)
    kmask = np.zeros((n_kspace_frame, *sim.shape), np.float32)

    for i, sp in enumerate(shot_pos):
        for ii, (kk, _) in enumerate(sp):
            mask = np.zeros(sim.shape, dtype=bool)
            mask[_get_slicer(shot_batches[i][ii])] = 1
            if sim.n_coils == 1:
                kdata[kk, 0, mask] = kdata_t[i][ii]
            else:
                kdata[kk, :, mask] = kdata_t[i][ii].T
            kmask[kk] += mask
    return kdata, kmask


def acq_noncartesian(
    sim: SimData,
    trajectory_gen: TrajectoryGeneratorType,
    n_shot_sim_frame: int,
    n_kspace_frame: int,
    n_jobs: int = -1,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Acquire with non cartesian stuff."""
    if sim.data_acq is None:
        raise ValueError("data acq not available")
    test_traj = next(trajectory_gen)
    n_samples = np.prod(test_traj.shape[:-1])
    dim = test_traj.shape[-1]

    logger.debug("extra kwargs %s", kwargs)
    op_kwargs = dict(
        shape=sim.shape,
        n_coils=sim.n_coils,
        density=False,
        smaps=sim.smaps,
        **kwargs,
    )

    scheduler = kspace_bulk_shot(trajectory_gen, sim.n_frames, n_shot_sim_frame)
    shot_batches, shot_pos, kdata_t = zip(
        *Parallel(
            n_jobs=n_jobs,
            backend="loky",
        )(
            delayed(_nufft_worker)(sim_frame, shot_batch, shot_pos, op_kwargs)
            for sim_frame, shot_batch, shot_pos in tqdm(
                work_generator(sim.data_acq, scheduler)
            )
        )
    )

    get_reusable_executor().shutdown(wait=True)

    # reorganize the data
    kdata = np.zeros((n_kspace_frame, sim.n_coils, n_samples), np.complex64)
    kmask = np.zeros((n_kspace_frame, n_samples, dim), np.float32)
    L = shot_batches[0].shape[1]
    for i, sp in enumerate(shot_pos):
        for ii, (kk, ss) in enumerate(sp):
            kdata[kk, :, ss * L : (ss + 1) * L] = kdata_t[i][..., ii * L : (ii + 1) * L]
            kmask[kk, ss * L : (ss + 1) * L] = shot_batches[i][ii]
    return kdata, kmask


def work_generator(
    data_acq: np.ndarray | LazySimArray, kspace_bulk_gen: Generator
) -> Generator[tuple, None, None]:
    """Set up all the work."""
    with DuplicateFilter(logging.getLogger("simulation")):
        for sim_frame_idx, shot_batch, shot_pos in kspace_bulk_gen:
            sim_frame = np.complex64(data_acq[sim_frame_idx])  # heavy to compute
            yield sim_frame, shot_batch, shot_pos


def _nufft_worker(
    sim_frame: np.ndarray,
    shot_batch: NDArray[np.int_],
    shot_pos: tuple[int, int],
    op_kwargs: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """Perform a shot acquisition."""
    import cupy as cp

    cp.arange(50).get()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="mrinufft",
        )

        fourier_op = get_operator(samples=shot_batch, **op_kwargs)
        kspace = fourier_op.op(sim_frame)
    #  write to share memory shots location and values.
    return shot_batch, shot_pos, kspace


def _cart_worker(
    sim_frame: np.ndarray,
    shot_batch: NDArray[np.int_],
    shot_pos: tuple[int, int],
    **kwargs: Mapping[str, Any],
) -> tuple[NDArray[np.int_], tuple[int, int], tuple[np.ndarray]]:

    masks = np.zeros((len(shot_batch), *sim_frame.shape), dtype=np.bool_)
    for i, shot in enumerate(shot_batch):
        masks[i][_get_slicer(shot)] = 1
    # ensure that mask are orthogonal !
    assert np.all(np.sum(masks, axis=0) < 2)
    mask = np.sum(masks, axis=0)
    fourier_op = FFT_Sense(sim_frame.shape, mask=mask, **kwargs)

    process_kspace: np.ndarray = fourier_op.op(sim_frame)
    # split to mask
    if fourier_op.n_coils == 1:
        kspaces = tuple(process_kspace[m == 1] for m in masks)
    else:
        kspaces = tuple(process_kspace[:, m == 1] for m in masks)
    return shot_batch, shot_pos, kspaces
