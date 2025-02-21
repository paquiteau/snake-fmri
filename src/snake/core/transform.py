"""Mathematical transformations of data."""

import logging
from collections.abc import Callable
from functools import partial
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from snake._meta import ThreeInts

from .parallel import run_parallel

log = logging.getLogger(__name__)


def effective_affine(old: NDArray, new: NDArray) -> NDArray:
    """Compute the effective affine transformation between two affine matrices."""
    new_affine = np.asarray(new, dtype=np.float32)
    old_affine = np.asarray(old, dtype=np.float32)
    return np.linalg.inv(new_affine) @ old_affine


def _validate_gpu_affine(use_gpu: bool = True) -> tuple[bool, Callable, ModuleType]:
    """Check if we can use the affine_transform from cupy."""
    if use_gpu:
        try:
            import cupy as xp
            from cupyx.scipy.ndimage import affine_transform as cu_affine_transform

            try:
                use_gpu = xp.cuda.is_available()
            except Exception as exc:
                use_gpu = False
                raise ImportError from exc

            affine_transform = partial(cu_affine_transform, texture_memory=True)
        except ImportError:
            use_gpu = False
    if not use_gpu:
        import numpy as xp

        log.warning("Cupy not available, using CPU.")
        from scipy.ndimage import affine_transform
    return use_gpu, affine_transform, xp


def apply_affine(
    data: NDArray[np.float32],
    old_affine: NDArray[np.float32],
    new_affine: NDArray[np.float32],
    new_shape: ThreeInts,
    transform_affine: NDArray[np.float32] = None,
    use_gpu: bool = True,
) -> NDArray[np.float32]:
    """Apply the new affine on the data.

    Parameters
    ----------
    data : NDArray
        Data to be transformed. 3D Array.
    old_affine : NDArray
        Affine of the original data
    new_affine : NDArray
        Affine of the new data
    new_shape : ThreeInts
        Shape of the new data
    transform_affine : NDArray, optional
        Transformation affine, by default None
    use_gpu : bool, optional

    Returns
    -------
    NDArray
    Resampled data with ``new_affine`` orientation and ``new_shape`` shape.
    """
    use_gpu, affine_transform, xp = _validate_gpu_affine(use_gpu)
    if transform_affine is None:
        transform_affine = effective_affine(new_affine, old_affine)
    transform_affine = xp.asarray(transform_affine, dtype=xp.float32)

    new_data = xp.zeros(new_shape, dtype=xp.float32)
    affine_transform(data, transform_affine, output_shape=new_shape, output=new_data)
    if use_gpu:
        new_data = new_data.get()

    return new_data


def apply_affine4d(
    data: NDArray,
    old_affine: NDArray,
    new_affine: NDArray,
    new_shape: ThreeInts,
    use_gpu: bool = False,
    n_jobs: int = -1,
    axis: int = 0,
) -> NDArray:
    """
    Apply the new affine on 4D data.

    Parameters
    ----------
    data : NDArray
        Data to be transformed. 3D Array.
    old_affine : NDArray
        Affine of the original data
    new_affine : NDArray
        Affine of the new data
    new_shape : ThreeInts
        Shape of the new data
    transform_affine : NDArray, optional
        Transformation affine, by default None
    use_gpu : bool, optional

    Returns
    -------
    NDArray
        Resampled data with ``new_affine`` orientation and ``new_shape`` shape.

    See Also
    --------
    apply_affine
    """
    n_frames = data.shape[axis]

    new_array = np.zeros((n_frames, *new_shape), dtype=np.float32)

    transform_affine = effective_affine(new_affine, old_affine)
    use_gpu = _validate_gpu_affine(use_gpu)[0]

    run_parallel(
        apply_affine,
        data,
        new_array,
        use_gpu=use_gpu,
        transform_affine=transform_affine,
        new_shape=new_shape,
        n_jobs=n_jobs,
        parallel_axis=axis,
    )

    return new_array


def serialize_array(arr: NDArray) -> str:
    """Serialize the array for mrd compatible format."""
    return "__".join(
        [
            base64.b64encode(arr.tobytes()).decode(),
            str(arr.shape),
            str(arr.dtype),
        ]
    )


def unserialize_array(s: str) -> NDArray:
    """Unserialize the array for mrd compatible format."""
    data, shape, dtype = s.split("__")
    shape = eval(shape)  # FIXME
    return np.frombuffer(base64.b64decode(data.encode()), dtype=dtype).reshape(*shape)
