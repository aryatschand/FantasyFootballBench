#!/usr/bin/env python3
import os
import csv
import glob

HERE = os.path.dirname(os.path.abspath(__file__))

def read_week_totals(path):
    totals = []  # list of dicts: {week, team, opponent, points}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("Player") or "").strip().upper() == "TOTAL":
                try:
                    pts = float(row.get("Points", 0) or 0)
                except Exception:
                    pts = 0.0
                try:
                    week = int(row.get("Week", 0) or 0)
                except Exception:
                    week = 0
                totals.append({
                    "week": week,
                    "team": row.get("Team", ""),
                    "opponent": row.get("Opponent", ""),
                    "points": pts,
                })
    return totals

def main():
    week_files = sorted(glob.glob(os.path.join(HERE, "week_*_results.csv")))
    standings = {}  # team -> {wins, losses, ties, points_for}
    for wf in week_files:
        totals = read_week_totals(wf)
        # Index by team for quick lookup
        by_team = {(t["team"], t["opponent"]): t for t in totals}
        seen = set()
        for t in totals:
            team = t["team"]
            opp = t["opponent"]
            week = t["week"]
            if not team or not opp:
                continue
            key = (week, tuple(sorted([team, opp])))
            if key in seen:
                continue
            seen.add(key)

            a = by_team.get((team, opp))
            b = by_team.get((opp, team))
            if not a or not b:
                # Incomplete pair; skip scoring but still accumulate points_for if desired
                for tm, pts in [(team, a["points"] if a else 0.0), (opp, b["points"] if b else 0.0)]:
                    rec = standings.setdefault(tm, {"wins":0, "losses":0, "ties":0, "points_for":0.0})
                    rec["points_for"] += pts
                continue

            # Update points_for
            for tm, pts in [(team, a["points"]), (opp, b["points"])]:
                rec = standings.setdefault(tm, {"wins":0, "losses":0, "ties":0, "points_for":0.0})
                rec["points_for"] += pts

            # Decide win/loss/tie
            if a["points"] > b["points"]:
                standings[team]["wins"] += 1
                standings[opp]["losses"] += 1
            elif b["points"] > a["points"]:
                standings[opp]["wins"] += 1
                standings[team]["losses"] += 1
            else:
                standings[team]["ties"] += 1
                standings[opp]["ties"] += 1

    # Emit final standings
    rows = []
    for team, rec in standings.items():
        rows.append({
            "Team": team,
            "Wins": rec["wins"],
            "Losses": rec["losses"],
            "Ties": rec["ties"],
            "PointsFor": f"{rec['points_for']:.2f}",
        })
    # Sort by wins desc, then points_for desc
    rows.sort(key=lambda r: (r["Wins"], float(r["PointsFor"])), reverse=True)

    out_path = os.path.join(HERE, "final_standings_2024.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Team","Wins","Losses","Ties","PointsFor"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

if __name__ == "__main__":
    main()


