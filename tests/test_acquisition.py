#!/usr/bin/env python3
import pytest
from simfmri.simulation import SimData
from simfmri.handlers import (
    CartesianAcquisitionHandler,
    BigPhantomGeneratorHandler,
)


@pytest.fixture
def simulation():
    sim = SimData((48, 48), sim_tr=0.1, n_coils=1, n_frames=200)
    sim = BigPhantomGeneratorHandler()(sim)
    return sim