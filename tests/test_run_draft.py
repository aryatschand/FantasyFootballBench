#!/usr/bin/env python3
import os
import sys
import csv
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_draft import draft_simulation, ROSTER_SLOTS


def test_draft_outputs():
    # Run the draft simulation
    draft_simulation()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "data", "draft_results")
    files = glob.glob(os.path.join(out_dir, "*_2024.csv"))
    assert len(files) == 2, "Should produce two team CSVs"

    total_required = sum(c for _, c in ROSTER_SLOTS)

    for path in files:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Check roster size
            assert len(rows) == total_required, f"Roster size incorrect in {path}"
            # Check required columns exist
            for col in ["Name", "Team", "Position", "FantasyPoints"]:
                assert col in reader.fieldnames

    # Check no duplicate players across teams
    all_names = []
    for path in files:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_names.append(row["Name"])
    assert len(all_names) == len(set(all_names)), "Duplicate players drafted across teams"


