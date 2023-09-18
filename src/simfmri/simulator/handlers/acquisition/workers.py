"""Multiprocessing module for the acquisition of data."""
from __future__ import annotations
import logging
import warnings
from contextlib import nullcontext
from typing import ContextManager, Mapping, Any
from dataclasses import dataclass, field
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from simfmri.simulator.simulation import SimDataType

from fmri.operators.fourier import FFT_Sense
from tqdm.auto import tqdm

from .trajectory import (
    TrajectoryGeneratorType,
    kspace_bulk_shot,
)
from mrinufft import get_operator


KILLER = None  # explicit is better than implicit

# from mrinufft import get_operator

logger = logging.getLogger("simulation." + __name__)


@dataclass
class ArrayInfo:
    """Information about an array."""

    shape: tuple[int, ...]
    dtype: np.dtype
    has_lock: bool = True
    lock: mp.Lock | ContextManager = field(init=False)

    def __post_init__(self):
        if self.has_lock:
            self.lock = mp.Lock()
        else:
            self.lock = nullcontext()

    def sm2np(self, shared_mem: SharedMemory) -> np.ndarray:
        """Create a numpy array from a shared memory object."""
        return np.ndarray(self.shape, buffer=shared_mem.buf, dtype=self.dtype)

    @property
    def byte_size(self) -> int:
        """Return size of array in byte."""
        return np.prod(self.shape) * self.dtype().itemsize


class AcquisitionWorker(mp.Process):
    """A process performing a MRI acquisition."""

    def __init__(
        self,
        kdata_sm: SharedMemory,
        kmask_sm: SharedMemory,
        sim: SimDataType,
        job_queue: mp.Queue,
        kdata_info: ArrayInfo,
        kmask_info: ArrayInfo,
        **kwargs: Mapping[str, Any],
    ):
        super().__init__()
        self.kdata_sm = kdata_sm
        self.kmask_sm = kmask_sm
        self.kdata_info = kdata_info
        self.kmask_info = kmask_info
        self.job_queue = job_queue
        self.sim = sim
        self.kwargs = kwargs

    def run(self) -> None:
        """Run the job."""
        while True:
            job = self.job_queue.get()
            if job is KILLER:
                break
            self._run(**job)
            self.job_queue.task_done()
        self.job_queue.task_done()
        self.kdata_sm.close()
        self.kmask_sm.close()


def _get_slicer(shot: np.ndarray) -> tuple[slice, slice, slice]:
    """Return a slicer for the mask.

    Fully sampled axis are marked with a -1.
    """
    slicer = [slice(None, None, None)] * shot.shape[-1]
    accel_axis = [i for i, v in enumerate(shot[0]) if v != -1][0]
    slicer[accel_axis] = shot[0][accel_axis]
    return tuple(slicer)


class CartesianWorker(AcquisitionWorker):
    """A process performing a cartesian MRI acquisition."""

    def _run(
        self,
        sim_frame_idx: int,
        n_kspace_frames: int,
        shot_batch: np.ndarray,
        shot_pos: list[tuple(int)],
    ) -> None:
        """Acquire cartesian data with parallel processing. Single worker."""
        sim_frame = np.complex64(self.sim.data_acq[sim_frame_idx])

        masks = np.zeros((len(shot_batch), *sim_frame.shape), dtype=np.int8)
        for i, shot in enumerate(shot_batch):
            masks[i][_get_slicer(shot)] = 1
        mask = np.sum(masks, axis=0)
        fourier_op = FFT_Sense(
            sim_frame.shape, mask=mask, smaps=self.sim.smaps, n_coils=self.sim.n_coils
        )
        process_kspace = fourier_op.op(sim_frame)

        # TODO make the array in the context manager.
        kdata = self.kdata_info.sm2np(self.kdata_sm)
        kmask = self.kmask_info.sm2np(self.kmask_sm)

        for m, (k, _) in zip(masks, shot_pos):
            with self.kdata_info.lock:
                kdata[k, ...] += process_kspace * m
            with self.kmask_info.lock:
                kmask[k, ...] |= m


class NonCartesianWorker(AcquisitionWorker):
    """Acquire non cartesian data."""

    def _run(
        self,
        sim_frame_idx: int,
        n_kspace_frames: int,
        shot_batch: np.ndarray,
        shot_pos: list[tuple[int, int]],
    ) -> None:
        """Acquire cartesian data with parallel processing. Single worker."""
        sim_frame = np.complex64(self.sim.data_acq[sim_frame_idx])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Samples will be rescaled to .*",
                category=UserWarning,
                module="mrinufft",
            )
            fourier_op = get_operator(self.kwargs["backend"])(
                samples=shot_batch,
                shape=sim_frame.shape,
                smaps=self.sim.smaps,
                n_coils=self.sim.n_coils,
                density=False,
            )

        kspace = fourier_op.op(sim_frame)

        L = shot_batch.shape[1]

        kdata = self.kdata_info.sm2np(self.kdata_sm)
        kmask = self.kmask_info.sm2np(self.kmask_sm)
        for i, (k, s) in enumerate(shot_pos):
            with self.kdata_info.lock:
                kdata[k, :, s * L : (s + 1) * L] = kspace[:, :, i * L : (i + 1) * L]
            with self.kmask_info.lock:
                kmask[k, s * L : (s + 1) * L] = shot_batch[i]


def _acquire_mp(
    sim: SimDataType,
    trajectory_gen: TrajectoryGeneratorType,
    worker_klass: AcquisitionWorker,
    n_shot_sim_frame: int,
    n_kspace_frame: int,
    n_jobs: int,
    kdata_info: ArrayInfo,
    kmask_info: ArrayInfo,
    **kwargs: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Acquire cartesian data with parallel processing.

    n_jobs are launched to acquire the data.
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count() // 2

    logger.debug("Acquiring data with multiprocessing, n_jobs=%d", n_jobs)
    logger.debug(
        "Full memory estimate for simulation: %f GB (lazy=%s)",
        sim.memory_estimate("GB"),
        sim.lazy,
    )
    logger.debug(
        "Full memory  estimate for kspace %f GB",
        kdata_info.byte_size / 1024**3,
    )

    acquisition_director = kspace_bulk_shot(
        trajectory_gen, sim.n_frames, n_shot_sim_frame
    )

    work_queue = mp.JoinableQueue(maxsize=2 * n_jobs)

    workers = []

    kdata_sm = SharedMemory(size=kdata_info.byte_size, create=True)
    kmask_sm = SharedMemory(size=kmask_info.byte_size, create=True)
    # create worker
    for _ in range(n_jobs):
        w = worker_klass(
            kdata_sm,
            kmask_sm,
            sim,
            work_queue,
            kdata_info,
            kmask_info,
            **kwargs,
        )
        workers.append(w)
        logger.debug("Starting worker %s", w.name)
        w.start()

    # add jobs to queue (with max size )
    for sim_frame_idx, shot_batch, shot_pos in tqdm(
        acquisition_director, total=sim.n_frames
    ):
        work_queue.put(
            {
                "sim_frame_idx": sim_frame_idx,
                "shot_batch": shot_batch,
                "shot_pos": shot_pos,
                "n_kspace_frames": n_kspace_frame,
            }
        )
    work_queue.join()
    # cleanup
    for _ in workers:
        work_queue.put(KILLER)
    work_queue.join()
    del workers
    logger.debug("deleted all workers")
    return kdata_sm, kmask_sm


def acquire_cartesian_mp(
    sim: SimDataType,
    trajectory_gen: TrajectoryGeneratorType,
    n_shot_sim_frame: int,
    n_kspace_frame: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Acquire with cartesian stuff."""
    kdata_info = ArrayInfo((n_kspace_frame, sim.n_coils, *sim.shape), np.complex64)
    kmask_info = ArrayInfo((n_kspace_frame, *sim.shape), np.int8)

    kdata_sm, kmask_sm = _acquire_mp(
        sim,
        trajectory_gen,
        CartesianWorker,
        n_shot_sim_frame,
        n_kspace_frame,
        n_jobs,
        kdata_info,
        kmask_info,
    )

    kdata = kdata_info.sm2np(kdata_sm)
    kmask = kmask_info.sm2np(kmask_sm)

    kdata_sm.close()
    kmask_sm.close()
    kdata_sm.unlink()
    kmask_sm.unlink()

    return kdata, kmask


def acquire_noncartesian_mp(
    sim: SimDataType,
    trajectory_gen: TrajectoryGeneratorType,
    n_shot_sim_frame: int,
    n_kspace_frame: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Acquire with non cartesian stuff."""
    test_traj = next(trajectory_gen)
    n_samples = np.prod(test_traj.shape[:-1])
    dim = test_traj.shape[-1]

    kdata_info = ArrayInfo((n_kspace_frame, sim.n_coils, n_samples), np.complex64)
    kmask_info = ArrayInfo((n_kspace_frame, n_samples, dim), np.float32)

    kdata_sm, kmask_sm = _acquire_mp(
        sim,
        trajectory_gen,
        NonCartesianWorker,
        n_shot_sim_frame,
        n_kspace_frame,
        n_jobs,
        kdata_info,
        kmask_info,
        backend=sim.extra_infos["operator"],
    )

    kdata = kdata_info.sm2np(kdata_sm)
    kmask = kmask_info.sm2np(kmask_sm)

    return kdata, kmask
