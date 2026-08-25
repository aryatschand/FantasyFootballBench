#!/usr/bin/env python3
"""End-to-end draft check. Runs a real draft, so it is network-marked and slow."""

import csv
import glob
import os

import pytest

from ffbench.config import get_num_teams
from scripts.run_draft import ROSTER_SLOTS, draft_simulation

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set; skipping live draft simulation",
    ),
]


def test_draft_outputs(sim_root):
    draft_simulation()

    out_dir = os.path.join(sim_root, "draft_results")
    files = glob.glob(os.path.join(out_dir, "*_2024.csv"))
    files = [f for f in files if not os.path.basename(f).startswith("TopPlayers")]
    assert len(files) == get_num_teams(), f"Expected one CSV per team in {out_dir}"

    total_required = sum(count for _, count in ROSTER_SLOTS)
    all_names = []

    for path in files:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == total_required, f"Roster size incorrect in {path}"
            for col in ["Name", "Team", "Position", "FantasyPoints"]:
                assert col in reader.fieldnames, f"Missing column {col} in {path}"
            all_names.extend(row["Name"] for row in rows)

    assert len(all_names) == len(set(all_names)), "Duplicate players drafted across teams"
