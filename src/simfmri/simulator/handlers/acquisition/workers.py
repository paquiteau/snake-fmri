"""Multiprocessing module for the acquisition of data."""
from __future__ import annotations
import logging
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from simfmri.simulator.simulation import SimDataType

from fmri.operators.fourier import FFT_Sense
from tqdm.auto import tqdm

from .trajectory import (
    TrajectoryGeneratorType,
    kspace_bulk_shot,
)

KILLER = None  # explicit

# from mrinufft import get_operator

logger = logging.getLogger("simulation." + __name__)


def sm2np(
    shared_mem: SharedMemory, dtype: np.dtype, shape: tuple[int, ...]
) -> np.ndarray:
    """Create a numpy array from a shared memory object."""
    arr = np.frombuffer(shared_mem.buf, dtype=dtype)
    return arr.reshape(shape)


class AcquisitionWorker(mp.Process):
    """A process performing a MRI acquisition."""

    def __init__(
        self,
        kspace_data_sm: SharedMemory,
        kspace_mask_sm: SharedMemory,
        sim: SimDataType,
        job_queue: mp.Queue,
        lock_data: mp.Lock,
        lock_mask: mp.Lock,
    ):
        super().__init__()
        self.kspace_data_sm = kspace_data_sm
        self.kspace_mask_sm = kspace_mask_sm
        self.job_queue = job_queue
        self.sim = sim
        self._lock_data = lock_data
        self._lock_mask = lock_mask

    def run(self) -> None:
        """Run the job."""
        while True:
            job = self.job_queue.get()
            if job is KILLER:
                break
            self._run(**job)
            self.job_queue.task_done()
        self.job_queue.task_done()


class CartesianWorker(AcquisitionWorker):
    """A process performing a cartesian MRI acquisition."""

    @staticmethod
    def _get_slicer(shot: np.ndarray) -> tuple[slice, slice, slice]:
        """Return a slicer for the mask.

        Fully sampled axis are marked with a -1.
        """
        slicer = [slice(None, None, None)] * shot.shape[-1]
        accel_axis = [i for i, v in enumerate(shot[0]) if v != -1][0]
        slicer[accel_axis] = shot[0][accel_axis]
        return tuple(slicer)

    def _run(
        self,
        sim_frame_idx: int,
        n_kspace_frames: int,
        shot_batch: np.ndarray,
        shot_in_kspace_frames: np.ndarray,
    ) -> None:
        """Acquire cartesian data with parallel processing. Single worker."""
        sim_frame = np.complex64(self.sim.data_acq[sim_frame_idx])

        masks = np.zeros((len(shot_batch), *sim_frame.shape), dtype=np.int8)
        for i, shot in enumerate(shot_batch):
            masks[i][self._get_slicer(shot)] = 1
        mask = np.sum(masks, axis=0)
        fourier_op = FFT_Sense(
            sim_frame.shape, mask=mask, smaps=self.sim.smaps, n_coils=self.sim.n_coils
        )
        process_kspace = fourier_op.op(sim_frame)

        kspace_data = sm2np(
            self.kspace_data_sm,
            np.complex64,
            (n_kspace_frames, self.sim.n_coils, *sim_frame.shape),
        )
        kspace_mask = sm2np(
            self.kspace_mask_sm, np.int8, (n_kspace_frames, *sim_frame.shape)
        )

        for m, kspace_frame in zip(masks, shot_in_kspace_frames):
            with self._lock_data:
                kspace_data[kspace_frame, ...] += process_kspace * m
            with self._lock_mask:
                kspace_mask[kspace_frame, ...] |= m


class NonCartesianWorker(AcquisitionWorker):
    """Acquire non cartesian data."""

    def _run(
        self,
        sim_frame_idx: int,
        n_kspace_frames: int,
        shot_batch: np.ndarray,
        shot_in_kspace_frames: np.ndarray,
    ) -> None:
        """Acquire cartesian data with parallel processing. Single worker."""
        # sim_frame = np.complex64(self.sim.data_acq[sim_frame_idx])

        ...


def _acquire_mp(
    sim: SimDataType,
    trajectory_gen: TrajectoryGeneratorType,
    worker_klass: AcquisitionWorker,
    n_shot_sim_frame: int,
    n_kspace_frame: int,
    n_jobs: int,
    kspace_data_size: int,
    kspace_mask_size: int,
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
        kspace_data_size / 1024**3,
    )

    acquisition_director = kspace_bulk_shot(
        trajectory_gen, sim.n_frames, n_shot_sim_frame
    )

    work_queue = mp.JoinableQueue(maxsize=2 * n_jobs)

    workers = []

    with SharedMemoryManager() as smm:
        kspace_data_sm = smm.SharedMemory(size=kspace_data_size)
        kspace_mask_sm = smm.SharedMemory(size=kspace_mask_size)
        lock_data, lock_mask = mp.Lock(), mp.Lock()
        # create worker
        for _ in range(n_jobs):
            w = worker_klass(
                kspace_data_sm, kspace_mask_sm, sim, work_queue, lock_data, lock_mask
            )
            workers.append(w)
            logger.debug("Starting worker %s", w.name)
            w.start()

        # add jobs to queue (with max size )
        for sim_frame_idx, shot_batch, shot_in_kframe in tqdm(
            acquisition_director, total=sim.n_frames
        ):
            work_queue.put(
                {
                    "sim_frame_idx": sim_frame_idx,
                    "shot_batch": shot_batch,
                    "shot_in_kspace_frames": shot_in_kframe,
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
        return kspace_data_sm, kspace_mask_sm


def acquire_cartesian_mp(
    sim: SimDataType,
    trajectory_gen: TrajectoryGeneratorType,
    n_shot_sim_frame: int,
    n_kspace_frame: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Acquire with cartesian stuff."""
    kspace_data_size = (
        n_kspace_frame * np.prod(sim.shape) * sim.n_coils * np.complex64().itemsize
    )
    kspace_mask_size = n_kspace_frame * np.prod(sim.shape) * np.int8().itemsize

    kspace_data_sm, kspace_mask_sm = _acquire_mp(
        sim,
        trajectory_gen,
        n_shot_sim_frame,
        n_kspace_frame,
        n_jobs,
        kspace_data_size,
        kspace_mask_size,
    )

    kspace_data = sm2np(
        kspace_data_sm, np.complex64, (n_kspace_frame, sim.n_coils, *sim.shape)
    )
    kspace_mask = sm2np(kspace_mask_sm, np.int8, (n_kspace_frame, *sim.shape))

    return kspace_data, kspace_mask


def acquire_noncartesian_mp(
    sim: SimDataType,
    trajectory_gen: TrajectoryGeneratorType,
    n_shot_sim_frame: int,
    n_kspace_frame: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Acquire with non cartesian stuff."""
    kspace_data_size = (
        n_kspace_frame * np.prod(sim.shape) * sim.n_coils * np.complex64().itemsize
    )
    kspace_mask_size = sim.extra_infos[""] * np.float32().itemsize

    kspace_data_sm, kspace_mask_sm = _acquire_mp(
        sim,
        trajectory_gen,
        n_shot_sim_frame,
        n_kspace_frame,
        n_jobs,
        kspace_data_size,
        kspace_mask_size,
    )

    kspace_data = sm2np(
        kspace_data_sm, np.complex64, (n_kspace_frame, sim.n_coils, *sim.shape)
    )
    kspace_mask = sm2np(kspace_mask_sm, np.int8, (n_kspace_frame, *sim.shape))

    return kspace_data, kspace_mask
