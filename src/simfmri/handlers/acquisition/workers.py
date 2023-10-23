"""Multiprocessing module for the acquisition of data."""
from __future__ import annotations

from multiprocessing import shared_memory
import logging
import warnings
from typing import Any, Callable, Mapping, Generator
from joblib import Parallel, delayed
import numpy as np
from fmri.operators.fourier import FFT_Sense
from mrinufft import get_operator
from tqdm.auto import tqdm

from simfmri.simulation import SimData

from .trajectory import TrajectoryGeneratorType, kspace_bulk_shot

# from mrinufft import get_operator

logger = logging.getLogger("simulation." + __name__)


def _get_slicer(shot: np.ndarray) -> tuple[slice, slice, slice]:
    """Return a slicer for the mask.

    Fully sampled axis are marked with a -1.
    """
    slicer = [slice(None, None, None)] * shot.shape[-1]
    accel_axis = [i for i, v in enumerate(shot[0]) if v != -1][0]
    slicer[accel_axis] = shot[0][accel_axis]
    return tuple(slicer)


def _run_cartesian(
    sim: SimData,
    kdata: np.ndarray,
    kmask: np.ndarray,
    sim_frame_idx: int,
    shot_batch: np.ndarray,
    shot_pos: tuple[int, int],
    **kwargs: Mapping[str, Any],
) -> None:
    sim_frame = np.complex64(sim.data_acq[sim_frame_idx])

    masks = np.zeros((len(shot_batch), *sim_frame.shape), dtype=np.int8)
    for i, shot in enumerate(shot_batch):
        masks[i][_get_slicer(shot)] = 1
    mask = np.sum(masks, axis=0)
    fourier_op = FFT_Sense(
        sim_frame.shape, mask=mask, smaps=sim.smaps, n_coils=sim.n_coils
    )

    process_kspace = fourier_op.op(sim_frame)

    for m, (k, _) in zip(masks, shot_pos):
        kdata[k, ...] += process_kspace * m
        kmask[k, ...] |= m


def _run_noncartesian(
    sim: SimData,
    kdata: np.ndarray,
    kmask: np.ndarray,
    sim_frame_idx: int,
    shot_batch: np.ndarray,
    shot_pos: tuple[int, int],
    nufft_backend: str,
    **kwargs: Mapping[str, Any],
) -> None:
    sim_frame = np.complex64(sim.data_acq[sim_frame_idx])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Samples will be rescaled to .*",
            category=UserWarning,
            module="mrinufft",
        )
        fourier_op = get_operator(
            nufft_backend,
            samples=shot_batch,
            shape=sim.shape,
            smaps=sim.smaps,
            n_coils=sim.n_coils,
            density=False,
            **kwargs,
        )

    kspace = fourier_op.op(sim_frame)

    L = shot_batch.shape[1]

    for i, (k, s) in enumerate(shot_pos):
        kdata[k, :, s * L : (s + 1) * L] = kspace[..., i * L : (i + 1) * L]
        kmask[k, s * L : (s + 1) * L] = shot_batch[i]


def _acquire(
    sim: SimData,
    trajectory_gen: TrajectoryGeneratorType,
    runner: Callable,
    n_shot_sim_frame: int,
    n_kspace_frame: int,
    kdata_info: tuple[tuple[int, ...], np.dtype],
    kmask_info: tuple[tuple[int, ...], np.dtype],
    **kwargs: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    kdata = np.zeros(*kdata_info)
    kmask = np.zeros(*kmask_info)

    scheduler = kspace_bulk_shot(trajectory_gen, sim.n_frames, n_shot_sim_frame)

    for sim_frame_idx, shot_batch, shot_pos in tqdm(scheduler, total=sim.n_frames):
        runner(
            sim,
            kdata,
            kmask,
            sim_frame_idx,
            shot_batch,
            shot_pos,
            **kwargs,
        )

    return kdata, kmask


def acq_cartesian(
    sim: SimData,
    trajectory_gen: TrajectoryGeneratorType,
    n_shot_sim_frame: int,
    n_kspace_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Acquire with cartesian stuff."""
    kdata_info = ((n_kspace_frame, sim.n_coils, *sim.shape), np.complex64)
    kmask_info = ((n_kspace_frame, *sim.shape), np.int8)

    kdata, kmask = _acquire(
        sim,
        trajectory_gen,
        _run_cartesian,
        n_shot_sim_frame,
        n_kspace_frame,
        kdata_info,
        kmask_info,
    )

    return kdata, kmask


def acq_noncartesian(
    sim: SimData,
    trajectory_gen: TrajectoryGeneratorType,
    n_shot_sim_frame: int,
    n_kspace_frame: int,
    **kwargs: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Acquire with non cartesian stuff."""
    test_traj = next(trajectory_gen)
    n_samples = np.prod(test_traj.shape[:-1])
    dim = test_traj.shape[-1]

    kdata_infos = ((n_kspace_frame, sim.n_coils, n_samples), np.complex64)
    shm_kdata = shared_memory.SharedMemory(
        name="kdata",
        create=True,
        size=np.prod(kdata_infos[0]) * np.dtype(kdata_infos[1]).itemsize,
    )
    kmask_infos = ((n_kspace_frame, n_samples, dim), np.float32)
    shm_kmask = shared_memory.SharedMemory(
        name="kmask",
        create=True,
        size=np.prod(kmask_infos[0]) * np.dtype(kmask_infos[1]).itemsize,
    )

    nufft_backend = kwargs.pop("backend")
    logger.debug("Using backend %s", nufft_backend)
    kwargs["nufft_backend"] = nufft_backend
    if nufft_backend == "stacked":
        kwargs["z_index"] = "auto"
    logger.debug("extra kwargs %s", kwargs)

    smaps = sim.smaps
    op_kwargs = dict(
        shape=sim.shape,
        n_coils=sim.n_coils,
        density=False,
        backend_name=nufft_backend,
    )
    scheduler = kspace_bulk_shot(trajectory_gen, sim.n_frames, n_shot_sim_frame)
    Parallel(n_jobs=-1, backend="multiprocessing", mmap_mode="r")(
        delayed(_single_worker)(
            sim_frame, smaps, shot_batch, shot_pos, op_kwargs, kdata_infos, kmask_infos
        )
        for sim_frame, shot_batch, shot_pos in tqdm(work_generator(sim, scheduler))
    )

    kdata_ = np.ndarray(kdata_infos[0], buffer=shm_kdata.buf, dtype=kdata_infos[1])
    kmask_ = np.ndarray(kmask_infos[0], buffer=shm_kmask.buf, dtype=kmask_infos[1])

    kdata = np.copy(kdata_)
    kmask = np.copy(kmask_)
    del kdata_
    del kmask_

    shm_kdata.close()
    shm_kmask.close()
    shm_kdata.unlink()
    shm_kmask.unlink()

    return kdata, kmask


def work_generator(sim: SimData, kspace_bulk_gen: Generator) -> Generator[tuple]:
    """Setup all the work."""
    for sim_frame_idx, shot_batch, shot_pos in kspace_bulk_gen:
        sim_frame = np.complex64(sim.data_acq[sim_frame_idx])  # heavy to compute
        yield sim_frame, shot_batch, shot_pos


def _single_worker(
    sim_frame: np.ndarray,
    smaps: np.ndarray,
    shot_batch: np.ndarray,
    shot_pos: tuple[int, int],
    op_kwargs: Mapping[str, Any],
    kdata_infos: tuple[tuple[int], np.Dtype],
    kmask_infos: tuple[tuple[int], np.Dtype],
) -> None:
    """Perform a shot acquisition."""
    fourier_op = get_operator(samples=shot_batch, smaps=smaps, **op_kwargs)
    kspace = fourier_op.op(sim_frame)
    L = shot_batch.shape[1]

    shm_kdata = shared_memory.SharedMemory(name="kdata", create=False)
    shm_kmask = shared_memory.SharedMemory(name="kmask", create=False)

    kdata_ = np.ndarray(kdata_infos[0], buffer=shm_kdata.buf, dtype=kdata_infos[1])
    kmask_ = np.ndarray(kmask_infos[0], buffer=shm_kmask.buf, dtype=kmask_infos[1])

    for i, (k, s) in enumerate(shot_pos):
        kdata_[k, :, s * L : (s + 1) * L] = kspace[..., i * L : (i + 1) * L]
        kmask_[k, s * L : (s + 1) * L] = shot_batch[i]

    del kdata_
    del kmask_
    shm_kdata.close()
    shm_kmask.close()
