"""Shared test fixtures and helpers.

Import paths are handled by `pythonpath = ["."]` in pyproject.toml, so test
modules can `from ffbench... import ...` without touching sys.path.
"""

import os

import pytest


def requires_api_key():
    """Skip marker for tests that make live OpenRouter calls."""
    return pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set; skipping live model call",
    )


@pytest.fixture(scope="session")
def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def sim_root(repo_root):
    """Output directory for the current simulation, matching the runner scripts."""
    sim_id = os.environ.get("FFBENCH_SIM_ID")
    if not sim_id:
        latest = os.path.join(repo_root, "data", "simulations", "latest_simulation_id.txt")
        if os.path.exists(latest):
            with open(latest) as f:
                sim_id = f.read().strip()
    if not sim_id:
        pytest.skip("No simulation id available (set FFBENCH_SIM_ID or run a simulation first)")
    return os.path.join(repo_root, "data", "simulations", sim_id)
