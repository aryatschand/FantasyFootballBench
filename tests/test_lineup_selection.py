#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.simulate_season import select_starting_lineup


def test_select_starting_lineup_counts():
    roster = [
        {"Name": "QB One", "Team": "X", "Position": "QB"},
        {"Name": "RB One", "Team": "X", "Position": "RB"},
        {"Name": "RB Two", "Team": "X", "Position": "RB"},
        {"Name": "WR One", "Team": "X", "Position": "WR"},
        {"Name": "WR Two", "Team": "X", "Position": "WR"},
        {"Name": "TE One", "Team": "X", "Position": "TE"},
        {"Name": "RB Three", "Team": "X", "Position": "RB"},  # FLEX
        {"Name": "WR Bench", "Team": "X", "Position": "WR"},
    ]
    starters, bench = select_starting_lineup(roster)
    assert len(starters) == 7
    # Validate slot counts
    assert sum(1 for p in starters if p["Position"] == "QB") == 1
    assert sum(1 for p in starters if p["Position"] == "RB") >= 2
    assert sum(1 for p in starters if p["Position"] == "WR") >= 2
    assert sum(1 for p in starters if p["Position"] == "TE") >= 1


