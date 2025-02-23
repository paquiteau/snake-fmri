from pytest_cases import parametrize_with_cases, parametrize, fixture
import numpy as np

from snake.core import Phantom, SimConfig
from numpy.testing import assert_allclose
from snake.core.transform import apply_affine, apply_affine4d


class CasesAffines:
    def case_identity(self):
        return np.eye(4)

    def case_random(self):
        affine = np.eye(4)
        affine[:3, :] = np.random.rand(3, 4)
        return affine


@fixture
def base_array():
    return np.random.rand(5, 20, 20, 20)


@parametrize_with_cases("old_affine", cases=CasesAffines)
@parametrize_with_cases("new_affine", cases=CasesAffines)
@parametrize(new_shape=[(20, 20, 20), (10, 10, 10), (5, 5, 5)])
@parametrize(use_gpu=[True, False])
def test_affine_4d(new_affine, old_affine, base_array, new_shape, use_gpu):
    """test that apply_affine4D works like a loop of apply_affine."""
    new_affine = np.asarray(new_affine, dtype=np.float32)
    old_affine = np.asarray(old_affine, dtype=np.float32)
    from4D = apply_affine4d(
        base_array, old_affine, new_affine, new_shape=new_shape, use_gpu=use_gpu
    )

    for i in range(base_array.shape[0]):
        assert_allclose(
            apply_affine(
                base_array[i], old_affine, new_affine, new_shape, use_gpu=use_gpu
            ),
            from4D[i],
        )


@parametrize_with_cases("old_affine", cases=CasesAffines)
@parametrize_with_cases("new_affine", cases=CasesAffines)
@parametrize(new_shape=[(20, 20, 20), (10, 10, 10), (5, 5, 5)])
def test_affine_4d(new_affine, old_affine, base_array, new_shape):
    """Test that gpu and cpu affine computations are equivalent."""
    new_affine = np.asarray(new_affine, dtype=np.float32)
    old_affine = np.asarray(old_affine, dtype=np.float32)
    from_cpu = apply_affine(
        base_array[0], old_affine, new_affine, new_shape, use_gpu=False
    )
    from_gpu = apply_affine(
        base_array[0], old_affine, new_affine, new_shape, use_gpu=True
    )
    assert_allclose(from_cpu, from_gpu)
