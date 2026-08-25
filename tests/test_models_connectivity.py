#!/usr/bin/env python3
"""Live connectivity check for every model listed in config.json."""

import os

import pytest

from ffbench.config import get_models
from ffbench.llm import LLMManager

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set; skipping live model calls",
    ),
]


@pytest.mark.parametrize("model", get_models(), ids=lambda m: m["id"])
def test_model_responds(model):
    resp = LLMManager().call_with_prompt(model["id"], "Return the word READY only.")
    assert isinstance(resp, str) and "ready" in resp.lower(), (
        f"{model['name']} ({model['id']}) did not respond READY. Got: {resp!r}"
    )
