from typing import Any, Mapping

import numpy as np
from np.typing import NDArray


class Acquisition:
    "Container for Acquisition Data"

    def __init__(self):
        self.kspace_data: NDArray = None
        self.kspace_mask: NDArray = None
        self.shot_order: NDArray = None
        self.is_cartesian: bool = None
        self.name = None

        self._shot_time: float = None
        self._shot_frame_map: NDArray = None
        self._group_n_shot: int = 1
        self._traj_name: str = ""
        self._traj_params: Mapping[str, Any] = None

    def to_mrd_data(self, dset):
        """Convert the k-space data to an MRD Acquisition."""

    @property
    def TR_ms(self): ...

    @property
    def n_kspace_frame(self): ...
