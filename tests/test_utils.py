"""Test utilities."""
import pytest
import numpy as np

from snkf.utils.phantom.big import _is_in_triangle, _is_in_triangle_mplt

from snkf.utils.utils import validate_rng, cplx_type


@pytest.mark.parametrize("triangle_check", [_is_in_triangle, _is_in_triangle_mplt])
def test_is_in_triangle(triangle_check):
    tri = np.array([(22, 8), (12, 55), (7, 19)], dtype=np.float32)
    pts = np.array([[15, 20], [1, 7], [7, 19]], dtype=np.float32)
    assert np.all(triangle_check(pts, tri[0], tri[1], tri[2]) == [True, False, True])


@pytest.mark.parametrize(
    ("_dtype", "_cdtype"),
    [
        ("float64", np.complex128),
        ("float32", np.complex64),
        ("float128", np.complex256),
        ("int16", np.complex64),
    ],
)
def test_cplx_type(_dtype, _cdtype):
    assert cplx_type(_dtype) == _cdtype


@pytest.mark.parametrize("rng", [1, None, np.random.default_rng()])
def test_validate_rng(rng):
    rng_ret = validate_rng(rng)
    assert isinstance(rng_ret, np.random.Generator)
