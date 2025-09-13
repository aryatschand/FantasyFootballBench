#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.simulate_season import validate_trade_proposal, apply_trade


def test_validate_trade_proposal_basic():
    t1 = {"roster": [
        {"Name": "Player A", "Team": "X", "Position": "RB"},
        {"Name": "Player B", "Team": "X", "Position": "WR"},
    ]}
    t2 = {"roster": [
        {"Name": "Player C", "Team": "Y", "Position": "RB"},
        {"Name": "Player D", "Team": "Y", "Position": "WR"},
    ]}
    ok, _ = validate_trade_proposal(t1, t2, ["Player A"], ["Player C"])  # 1-for-1
    assert ok

    ok, _ = validate_trade_proposal(t1, t2, ["Player A", "Player B"], ["Player C"])  # unequal
    assert not ok


def test_apply_trade_swaps_players():
    t1 = {"roster": [
        {"Name": "Player A", "Team": "X", "Position": "RB"},
        {"Name": "Player B", "Team": "X", "Position": "WR"},
    ]}
    t2 = {"roster": [
        {"Name": "Player C", "Team": "Y", "Position": "RB"},
        {"Name": "Player D", "Team": "Y", "Position": "WR"},
    ]}
    apply_trade(t1, t2, ["Player A"], ["Player C"])  # swap
    t1_names = {p["Name"] for p in t1["roster"]}
    t2_names = {p["Name"] for p in t2["roster"]}
    assert "Player A" not in t1_names and "Player C" in t1_names
    assert "Player C" not in t2_names and "Player A" in t2_names


