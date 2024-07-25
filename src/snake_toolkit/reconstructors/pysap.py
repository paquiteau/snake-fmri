"""Reconstructors using PySAP-fMRI toolbox."""

import numpy as np
from numpy.typing import NDArray
from snake.simulation import SimConfig
from snake.mrd_utils import (
    CartesianFrameDataLoader,
    NonCartesianFrameDataLoader,
    MRDLoader,
)
from tqdm.auto import tqdm

from .base import BaseReconstructor
from .fourier import fft, ifft

PYSAP_FMRI_AVAILABLE = True
try:
    import fmri
except ImportError:
    PYSAP_FMRI_AVAILABLE = False


class ZeroFilledReconstructor(BaseReconstructor):
    """Zero Filled Reconstructor."""

    __reconstructor_name__ = "adjoint"

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

        final_images = np.zeros(
            (data_loader.n_frames, *data_loader.shape), dtype=np.float32
        )
        pbar = tqdm(total=data_loader.n_frames, desc="Reconstructing")
        for i, (_, kspace) in enumerate(data_loader):
            ...
            adj_data = ifft(kspace, axis=tuple(range(len(sim_conf.shape), 0, -1)))
            if smaps is not None and data_loader.n_coils > 1:
                adj_data_smaps_comb = np.sum(
                    abs(adj_data * smaps.conj()), axis=0
                ).astype(np.float32, copy=False)
            else:
                adj_data_smaps_comb = np.sum(abs(adj_data) ** 2, axis=0).astype(
                    np.float32, copy=False
                )
            final_images[i] = adj_data_smaps_comb
            pbar.update(1)
        pbar.close()
        return final_images

    def _reconstruct_nufft(
        self, data_loader: NonCartesianFrameDataLoader, sim_conf: SimConfig
    ) -> NDArray:
        """Reconstruct data with nufft method."""
        raise NotImplementedError("Missing nufft code.")


# EOF
