#!/usr/bin/env python

import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
import numpy as np
import numpy.testing as npt
from snake.core import FOVConfig


class CasesFOVConfig:
    def case_standard(self):
        return FOVConfig(
            angles=(0, 0, 0), offset=(0, 0, 0), res_mm=(1, 1, 1), size=(192, 192, 128)
        )

    @parametrize(angles=[(5, 0, 0), (0, 15, 0), (0, 0, 4), (3, 3, 0)])
    def case_rotation(self, angles):
        return FOVConfig(
            angles=angles, offset=(0, 0, 0), res_mm=(3, 3, 3), size=(192, 192, 128)
        )

    @parametrize(scale=[(1, 1, 1), (2, 2, 2), (0.5, 0.5, 0.5), (1, 2, 3)])
    def case_scaling(self, scale):
        return FOVConfig(
            angles=(0, 0, 0), offset=(0, 0, 0), res_mm=scale, size=(192, 192, 128)
        )

    @parametrize(offset=[(1, 1, 1), (2, 2, 2), (0.5, 0.5, 0.5), (1, 2, 3)])
    def case_scaling(self, offset):
        return FOVConfig(
            angles=(0, 0, 0), offset=offset, res_mm=(1, 1, 1), size=(192, 192, 128)
        )


@parametrize_with_cases("fov_config", cases=CasesFOVConfig)
def test_fov_affine(fov_config: FOVConfig):
    affine = fov_config.affine
    fov2 = FOVConfig.from_affine(affine, fov_config.size)
    npt.assert_allclose(fov_config.angles, fov2.angles, atol=1e-8)
    npt.assert_allclose(fov_config.offset, fov2.offset)
    npt.assert_allclose(fov_config.res_mm, fov2.res_mm)
    npt.assert_allclose(fov_config.size, fov2.size)
