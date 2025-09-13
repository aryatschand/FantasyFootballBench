#!/usr/bin/env python3
import os
import sys
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffbench.data_handler import DataHandler
from ffbench.scoring import calculate_fantasy_points
from ffbench.config import get_scoring


def normalize_stat_keys(row):
    return {
        "passing_yards": row.get("PassingYards", 0) or 0,
        "passing_tds": row.get("PassingTouchdowns", 0) or 0,
        "interceptions": row.get("PassingInterceptions", 0) or 0,
        "rushing_yards": row.get("RushingYards", 0) or 0,
        "rushing_tds": row.get("RushingTouchdowns", 0) or 0,
        "receptions": row.get("Receptions", 0) or 0,
        "receiving_yards": row.get("ReceivingYards", 0) or 0,
        "receiving_tds": row.get("ReceivingTouchdowns", 0) or 0,
        "fumbles_lost": row.get("FumblesLost", 0) or 0,
    }


def resolve_full_name_from_data(name, dh):
    """Resolve abbreviated name to full name using 2024 weekly data."""
    if " " in name and "." not in name:
        return name  # Already full name

    # Use weekly data as source of truth for full names
    df = dh.read_player_game_stats_by_week(2024, 1)  # Use week 1 as reference
    if not df.empty:
        # Look for exact match first (unlikely for abbreviated names)
        exact = df[df["Name"] == name]
        if not exact.empty:
            return name

        # Try to match abbreviation pattern
        if "." in name:
            parts = name.split(".")
            if len(parts) >= 2:
                first_initial = parts[0].strip()
                last_part = parts[1].strip()
                for _, row in df.iterrows():
                    full = row.get("Name", "")
                    if full and " " in full:
                        full_parts = full.split(" ")
                        if (len(full_parts) >= 2 and
                            full_parts[0].startswith(first_initial) and
                            (last_part in full_parts[-1] or full_parts[-1].startswith(last_part))):
                            return full

    return name  # Return original if can't resolve


def export_top_players_2024():
    dh = DataHandler()
    scoring = get_scoring()
    df = dh.read_player_season_projection_stats(2024)
    if df.empty:
        raise RuntimeError("No 2024 PlayerSeasonProjectionStats available.")

    rows = []
    for _, row in df.iterrows():
        pos = row.get("Position")
        if pos not in ("QB", "RB", "WR", "TE"):
            continue
        stats_map = normalize_stat_keys(row)
        pts = calculate_fantasy_points(stats_map, scoring)
        # Always use full name
        full_name = resolve_full_name_from_data(row.get("Name"), dh)
        rows.append({
            "Name": full_name,
            "Team": row.get("Team"),
            "Position": pos,
            "FantasyPoints": float(f"{pts:.2f}")
        })

    rows.sort(key=lambda r: r["FantasyPoints"], reverse=True)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "draft_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "TopPlayers_2024_Projections_PPR.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Name", "Team", "Position", "FantasyPoints"])
        for i, r in enumerate(rows, 1):
            writer.writerow([i, r["Name"], r["Team"], r["Position"], f"{r['FantasyPoints']:.2f}"])

    print(f"Saved top players to {out_path}")


if __name__ == "__main__":
    export_top_players_2024()


