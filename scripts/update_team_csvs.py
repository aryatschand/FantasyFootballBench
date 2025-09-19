#!/usr/bin/env python3
import os
import csv
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffbench.config import get_num_teams

def update_team_csvs():
    # Model ARN used in the draft (from the logs)
    model_arn = "arn:aws:bedrock:us-east-2:851725383897:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"

    draft_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "draft_results")

    num_teams = get_num_teams()
    for team_num in range(1, num_teams + 1):
        old_filename = f"Team_{team_num}_2024.csv"
        old_path = os.path.join(draft_dir, old_filename)

        if not os.path.exists(old_path):
            continue

        # Read existing CSV
        rows = []
        with open(old_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        # Create new filename with model ARN
        model_short = model_arn.split('/')[-1]  # Get just the model identifier
        new_filename = f"Team_{team_num}_{model_short}_2024.csv"
        new_path = os.path.join(draft_dir, new_filename)

        # Write new CSV with Model column
        with open(new_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Team", "Position", "FantasyPoints", "Model"])
            for row in rows:
                writer.writerow([
                    row["Name"],
                    row["Team"],
                    row["Position"],
                    row["FantasyPoints"],
                    model_arn
                ])

        # Remove old file
        os.remove(old_path)
        print(f"Updated {old_filename} -> {new_filename}")

if __name__ == "__main__":
    update_team_csvs()
