"""Reconstructors using PySAP-fMRI toolbox."""

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Local imports
from snake.simulation import SimConfig
from snake.mrd_utils import (
    CartesianFrameDataLoader,
    MRDLoader,
    NonCartesianFrameDataLoader,
)
from snake.parallel import array_from_shm, array_to_shm, SharedMemoryManager, ArrayProps

from .base import BaseReconstructor
from .fourier import ifft

PYSAP_FMRI_AVAILABLE = True
try:
    import fmri
except ImportError:
    PYSAP_FMRI_AVAILABLE = False


def _reconstruct_cartesian_frame(
    filename: str,
    idx: int,
    smaps_props: ArrayProps,
    final_props: ArrayProps,
) -> int:
    """Reconstruct a single frame."""
    data_loader = CartesianFrameDataLoader(filename)
    mask, kspace = data_loader.get_kspace_frame(idx)
    sim_conf = data_loader.get_sim_conf()
    with array_from_shm(final_props) as final_images:
        adj_data = ifft(kspace, axis=tuple(range(len(sim_conf.shape), 0, -1)))
        if smaps_props is not None and data_loader.n_coils > 1:
            with array_from_shm(smaps_props) as smaps:
                adj_data_smaps_comb = np.sum(
                    abs(adj_data * smaps.conj()), axis=0
                ).astype(np.float32, copy=False)
        else:
            adj_data_smaps_comb = np.sum(abs(adj_data) ** 2, axis=0).astype(
                np.float32, copy=False
            )
        final_images[0][idx] = adj_data_smaps_comb
    return idx


class ZeroFilledReconstructor(BaseReconstructor):
    """Zero Filled Reconstructor."""

    __reconstructor_name__ = "adjoint"
    n_jobs: int = 10

    def setup(self, sim_conf: SimConfig) -> None:
        """Initialize Reconstructor."""
        pass

    def reconstruct(self, data_loader: MRDLoader, sim_conf: SimConfig) -> NDArray:
        """Reconstruct data with zero-filled method."""
        if isinstance(data_loader, CartesianFrameDataLoader):
            return self._reconstruct_cartesian(data_loader, sim_conf)
        elif isinstance(data_loader, NonCartesianFrameDataLoader):
            return self._reconstruct_nufft(data_loader, sim_conf)
        else:
            raise ValueError("Unknown dataloader")

    def _reconstruct_cartesian(
        self, data_loader: CartesianFrameDataLoader, sim_conf: SimConfig
    ) -> NDArray:
        smaps = data_loader.get_smaps()
        if smaps is None and data_loader.n_coils > 1:
            raise NotImplementedError("Missing coil combine code.")

        final_images = np.ones(
            (data_loader.n_frames, *data_loader.shape), dtype=np.float32
        )

        with (
            SharedMemoryManager() as smm,
            ProcessPoolExecutor(self.n_jobs) as executor,
            tqdm(total=data_loader.n_frames) as pbar,
        ):
            smaps_props = None
            if smaps is not None:
                smaps_props, smaps_shared, smaps_sm = array_to_shm(smaps, smm)
            final_props, final_shared, final_sm = array_to_shm(final_images, smm)

            futures = {
                executor.submit(
                    _reconstruct_cartesian_frame,
                    data_loader.filename,
                    idx,
                    smaps_props,
                    final_props,
                ): idx
                for idx in range(data_loader.n_frames)
            }

            for future in as_completed(futures):
                future.result()
                pbar.update(1)
            final_images[:] = final_shared.copy()
            final_sm.close()
            if smaps_props is not None:
                smaps_sm.close()
            smm.shutdown()
        return final_images

    def _reconstruct_nufft(
        self, data_loader: NonCartesianFrameDataLoader, sim_conf: SimConfig
    ) -> NDArray:
        """Reconstruct data with nufft method."""
        raise NotImplementedError("Missing nufft code.")


# EOF
