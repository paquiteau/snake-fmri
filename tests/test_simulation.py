"""Test for simulation dataclass."""
import pytest
from simfmri.simulator import Simulation


@pytest.fixture
def sim():
    """Simple simulation object."""
    return Simulation(shape=(16, 16), n_frames=10, TR=1.0)


def test_simulation_duration(sim):
    """Test simulation duration declaration."""
    assert sim.duration == sim.TR * sim.n_frames
