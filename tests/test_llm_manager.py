#!/usr/bin/env python3
"""Smoke tests for LLMManager: prompt calls and signature-driven JSON calls."""

import os

import pytest

from ffbench.config import get_model_ids
from ffbench.llm import (
    ChooseStartersSignature,
    DraftPickSignature,
    LLMManager,
)

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set; skipping live model calls",
    ),
]


@pytest.fixture(scope="module")
def model_id():
    return get_model_ids()[0]


def test_call_with_prompt(model_id):
    response = LLMManager().call_with_prompt(model_id, "What is 2 + 2? Respond with just the number.")
    assert isinstance(response, str)
    assert "4" in response


def test_draft_pick_signature(model_id):
    result = LLMManager().call_with_signature(
        model_id=model_id,
        signature_class=DraftPickSignature,
        team_info="You are managing a fantasy football team. You need a QB.",
        available_players="Josh Allen (QB), Patrick Mahomes (QB), Joe Burrow (QB)",
    )
    assert isinstance(result, dict), f"Expected parsed JSON, got {type(result).__name__}"
    assert result.get("draft_pick"), f"No draft_pick in response: {result}"


def test_choose_starters_signature(model_id):
    result = LLMManager().call_with_signature(
        model_id=model_id,
        signature_class=ChooseStartersSignature,
        team_info=(
            "Your team has: Josh Allen (QB), Christian McCaffrey (RB), "
            "Davante Adams (WR), Travis Kelce (TE)"
        ),
    )
    assert isinstance(result, dict), f"Expected parsed JSON, got {type(result).__name__}"
    assert result.get("starter_lineup"), f"No starter_lineup in response: {result}"
