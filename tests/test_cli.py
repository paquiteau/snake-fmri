"""Test the CLI with simple scenarios."""

from hydra.test_utils.test_utils import run_python_script


def test_scenario1():
    cmd = [
        "src/snkf/cli/main.py",
        "--config-name=scenario1.yaml",
        "++handlers.acquisition-vds.n_jobs=1",  # to avoid to much memory consumption.
    ]
    result, _err = run_python_script(cmd, allow_warnings=True)
