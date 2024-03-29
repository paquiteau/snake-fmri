"""Test the CLI utils."""

import pytest
from omegaconf import OmegaConf
from snkf.cli.utils import hash_config


@pytest.fixture()
def config():
    return OmegaConf.create(
        {
            "test": {
                "pattern": "to_remove",
                "option1": 1,
            },
            "pattern2": {
                "pattern": "to_remove",
                "option2": "to_remove",
            },
        }
    )


@pytest.fixture()
def config2():
    return OmegaConf.create(
        {
            "test": {
                "option1": 1,
            },
        }
    )


def test_hash_config(config, config2):

    hash_val = hash_config(config, "pattern")

    hash_val2 = hash_config(config2)

    assert hash_val == hash_val2
