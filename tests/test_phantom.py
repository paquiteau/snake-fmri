"""Test on the phantom object."""

from pathlib import Path
from multiprocessing.managers import SharedMemoryManager
from pytest_cases import parametrize_with_cases, parametrize, fixture
import numpy as np
from snake.core import Phantom, SimConfig


@fixture(scope="module")
def sim_config() -> SimConfig:
    return SimConfig(
        max_sim_time=10,
    )


class CasesPhantom:
    """Test cases for the Phantom object."""

    @parametrize(n_coils=(1, 8))
    def case_mini_phantom(self, n_coils: int) -> Phantom:
        affine = np.eye(4)
        affine[:3, :] = np.random.rand(3, 4)
        phantom = Phantom(
            "dummy",
            masks=np.random.uniform(0, 1, (5, 10, 10, 10)),
            labels=np.array(["A", "B", "C", "D", "E"]),
            props=np.random.rand(5, 5),
            smaps=None if n_coils == 1 else np.random.rand(n_coils, 10, 10, 10),
            affine=affine,
        )
        return phantom


@parametrize_with_cases(
    "phantom",
    cases=CasesPhantom,
)
def test_multiprocessing(phantom: Phantom):
    """Check that the shared memory serialization works."""
    with SharedMemoryManager() as smm:
        array_props, _ = phantom.in_shared_memory(smm)
        with Phantom.from_shared_memory(*array_props) as phantom2:
            assert phantom.name == phantom2.name
            assert np.all(phantom.masks == phantom2.masks)
            assert np.all(phantom.labels == phantom2.labels)
            assert np.all(phantom.props == phantom2.props)
            assert np.all(phantom.smaps == phantom2.smaps)
            assert np.all(phantom.affine == phantom2.affine)


@parametrize_with_cases(
    "phantom",
    cases=CasesPhantom,
)
def test_mrd(phantom: Phantom, tmpdir: Path):
    """Test that the phantom is correctly written and red back"""

    dataset = tmpdir / "phantom.mrd"

    phantom.to_mrd_dataset(dataset)
    phantom2 = Phantom.from_mrd_dataset(dataset)
    print(phantom2.affine)
    print(phantom.affine)
    assert phantom.name == phantom2.name
    assert np.all(phantom.masks == phantom2.masks)
    assert np.all(phantom.labels == phantom2.labels)
    assert np.all(phantom.props == phantom2.props)
    assert np.all(phantom.smaps == phantom2.smaps)
    assert np.allclose(phantom.affine, phantom2.affine)


@parametrize_with_cases("phantom", cases=CasesPhantom)
@parametrize("use_gpu", [True, False])
def test_contrast(phantom, sim_config, use_gpu):
    """Test that the phantom can be used in a simulation."""
    contrast = phantom.contrast(
        sim_conf=sim_config,
    )
    # FIXME: This is not the correct way to test the contrast
    assert contrast.shape == sim_config.fov.shape
