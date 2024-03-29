"""Test for simulation dataclass."""


def test_simulation_params():
    """Test the simulation params."""
    from snkf.simulation import SimParams

    sim_params = SimParams(shape=(48, 48, 32), sim_time=12, sim_tr=1.0)
    assert sim_params.shape == (48, 48, 32)
    assert sim_params.sim_tr == 1.0
    assert sim_params.n_coils == 1
    assert sim_params.rng == 19980408
    assert sim_params.n_frames == 12
    assert sim_params.extra_infos == dict()


def test_simulation_data():
    """Test the simulation data."""
    from snkf.simulation import SimData

    sim_data = SimData.from_params(
        shape=(48, 48, 32), sim_time=12, sim_tr=1.0, fov=(0.1,) * 3
    )
    assert sim_data.shape == (48, 48, 32)
    assert sim_data.n_frames == 12
    assert sim_data.sim_tr == 1.0
    assert sim_data.n_coils == 1
    assert sim_data.rng == 19980408
    assert sim_data.extra_infos == dict()
    assert sim_data.is_valid() is True
