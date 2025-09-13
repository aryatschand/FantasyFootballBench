#!/usr/bin/env python3
import os
import sys
import random
import csv
import json
from datetime import datetime
from collections import defaultdict, Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffbench.data_handler import DataHandler
from ffbench.scoring import calculate_fantasy_points
from ffbench.llm import LLMManager, DraftPickSignature
from ffbench.config import get_scoring, get_roster_slots, get_models, get_num_teams, format_roster_format, format_scoring_format


ROSTER_SLOTS = get_roster_slots()

SCORING = get_scoring()


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


def is_flex_eligible(position):
    return position in ("RB", "WR", "TE")


def build_full_name_map_2024(dh):
    """Build a map from PlayerID -> full Name using 2024 weekly data."""
    id_to_name = {}
    for wk in range(1, 18):
        df = dh.read_player_game_stats_by_week(2024, wk)
        if df.empty:
            continue
        for _, row in df.iterrows():
            pid = row.get("PlayerID")
            nm = row.get("Name")
            if pid and isinstance(nm, str) and " " in nm:
                id_to_name[pid] = nm
    return id_to_name


def build_player_pool_2024(dh):
    season_df = dh.read_player_season_stats(2024)
    if season_df.empty:
        raise RuntimeError("No 2024 PlayerSeasonStats available.")
    # Resolve names to full names via weekly mapping
    id_to_full = build_full_name_map_2024(dh)
    # Calculate fantasy points for each player
    fantasy_points = []
    for _, row in season_df.iterrows():
        pos = row.get("Position")
        if pos not in ("QB", "RB", "WR", "TE"):
            continue
        stats_map = normalize_stat_keys(row)
        pts = calculate_fantasy_points(stats_map, SCORING)
        # Prefer full name from weekly mapping
        full_name = id_to_full.get(row.get("PlayerID"), row.get("Name"))
        fantasy_points.append({
            "Name": full_name,
            "Team": row.get("Team"),
            "Position": pos,
            "FantasyPoints": pts,
        })
    return fantasy_points


def top_remaining_by_position(player_pool, drafted_set, top_k=10):
    # Shuffle to avoid deterministic order before slicing
    random.shuffle(player_pool)
    pos_to = defaultdict(list)
    for p in player_pool:
        if p["Name"] in drafted_set:
            continue
        pos_to[p["Position"]].append(p)
    # sort by internal hidden ranking (fantasy points), but we already shuffled to hide strict best
    for pos in pos_to:
        pos_to[pos].sort(key=lambda x: x["FantasyPoints"], reverse=True)
        pos_to[pos] = pos_to[pos][:top_k]
    return pos_to


def load_top_players_csv():
    """Load precomputed top players CSV and return full ordered lists per position."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "draft_results", "TopPlayers_2024_Projections_PPR.csv")
    top_map = {"QB": [], "RB": [], "WR": [], "TE": []}  # list of (name, points)
    if not os.path.exists(path):
        return top_map
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos = row.get("Position")
            if pos in top_map:
                try:
                    pts = float(row.get("FantasyPoints", 0) or 0)
                except Exception:
                    pts = 0.0
                top_map[pos].append((row.get("Name"), pts))
    # Ensure sorted by points desc
    for pos in top_map:
        top_map[pos].sort(key=lambda x: x[1], reverse=True)
    return top_map


def top5_available_from_csv(precomputed_map, pos, drafted_set, roster_filter_fn, fallback_names):
    """Return up to 5 top-scoring available names for a position, respecting roster fit.
    If insufficient, fallback to supplied names list.
    """
    names = []
    for name, _ in precomputed_map.get(pos, []):
        if name in drafted_set:
            continue
        if roster_filter_fn(name):
            names.append(name)
        if len(names) >= 5:
            break
    if len(names) < 5:
        for name in fallback_names:
            if name in drafted_set or name in names:
                continue
            if roster_filter_fn(name):
                names.append(name)
            if len(names) >= 5:
                break
    # Shuffle within the 5 to avoid fixed order bias
    random.shuffle(names)
    return names


def roster_state_string(roster_slots_filled):
    parts = []
    for slot, need, have in roster_slots_filled:
        parts.append(f"{slot}: {have}/{need}")
    return ", ".join(parts)


# Removed resolve_full_name_from_weeks function - now using full names directly


def summarize_weekly_year(dh, full_name, year):
    total = {
        "PassingYards": 0.0,
        "PassingTouchdowns": 0.0,
        "PassingInterceptions": 0.0,
        "RushingYards": 0.0,
        "RushingTouchdowns": 0.0,
        "Receptions": 0.0,
        "ReceivingYards": 0.0,
        "ReceivingTouchdowns": 0.0,
    }
    found = False
    for week in range(1, 18):
        df = dh.read_player_game_stats_by_week(year, week)
        if df.empty:
            continue
        row = df[df["Name"] == full_name]
        if row.empty:
            continue
        found = True
        r = row.iloc[0]
        for k in total.keys():
            total[k] += float(r.get(k, 0) or 0)
    return found, total


def format_player_one_liner(dh, full_name):
    # full_name is already a full name, no resolution needed

    parts = []
    for year in (2022, 2023):
        found, totals = summarize_weekly_year(dh, full_name, year)
        if found:
            parts.append(
                f"{year}: {int(totals['PassingYards'])} pass yds, {int(totals['PassingTouchdowns'])} pass TDs, {int(totals['PassingInterceptions'])} INTs, "
                f"{int(totals['RushingYards'])} rush yds, {int(totals['RushingTouchdowns'])} rush TDs, {int(totals['Receptions'])} rec, {int(totals['ReceivingYards'])} rec yds, {int(totals['ReceivingTouchdowns'])} rec TDs"
            )
        else:
            parts.append(f"{year}: No data")

    # 2024 projection (week 1) by full name
    proj_df = dh.read_player_game_projection_stats_by_week(2024, 1)
    if not proj_df.empty:
        prow = proj_df[proj_df["Name"] == full_name]
        if not prow.empty:
            r = prow.iloc[0]
            parts.append(
                f"2024 proj: {int(r.get('PassingYards', 0) or 0)} pass yds, {int(r.get('PassingTouchdowns', 0) or 0)} pass TDs, {int(r.get('PassingInterceptions', 0) or 0)} INTs, "
                f"{int(r.get('RushingYards', 0) or 0)} rush yds"
            )
        else:
            parts.append("2024 proj: No projections available")
    else:
        parts.append("2024 proj: No projections available")

    return " | ".join(parts)


def draft_simulation():
    random.seed(42)
    dh = DataHandler()
    player_pool = build_player_pool_2024(dh)

    # Hidden ranking for internal use
    hidden_ranked = sorted(player_pool, key=lambda x: x["FantasyPoints"], reverse=True)

    # Two teams with two Bedrock models
    # Load models list from env (comma-separated). If not set, default to 10 instances of Claude Sonnet
    models = get_models()
    # Build teams based on models provided; include nickname in team name
    teams = [
        {"name": f"{models[i]['name']}_Team_{i+1}", "model": models[i], "picks": []}
        for i in range(len(models))
    ]

    # Roster constraints
    base_slots = [(slot, count) for slot, count in ROSTER_SLOTS]

    llm = LLMManager()
    drafted = set()

    # Helper to evaluate if player fits roster
    def can_add(team_picks, name):
        # Count picks by position + flex eligibility
        pos_counts = Counter([p["Position"] for p in team_picks])
        # build remaining requirements
        remaining = {slot: need for slot, need in base_slots}
        # Fill specific slots
        for pos, need in list(remaining.items()):
            filled = pos_counts[pos] if pos != "FLEX" else 0
            remaining[pos] = max(0, need - filled)

        # If position slot has space, allow
        pos = next((p["Position"] for p in player_pool if p["Name"] == name), None)
        if pos is None:
            return False
        if remaining.get(pos, 0) > 0:
            return True
        # Else check flex
        if is_flex_eligible(pos) and remaining.get("FLEX", 0) > 0:
            return True
        # Bench: allow any position if bench slots remain
        if remaining.get("BENCH", 0) > 0:
            return True
        return False

    # Helper to update roster state string
    def roster_slots_filled(team_picks):
        pos_counts = Counter([p["Position"] for p in team_picks])
        flex_count = len([p for p in team_picks if is_flex_eligible(p["Position"])]) - (
            pos_counts.get("RB", 0) + pos_counts.get("WR", 0) + pos_counts.get("TE", 0)
        )
        # Approximate: count flex as any overflow beyond base slots filled
        res = []
        for slot, need in base_slots:
            if slot == "FLEX":
                have = max(0, len([p for p in team_picks if is_flex_eligible(p["Position"])]) - (
                    ROSTER_SLOTS[1][1] + ROSTER_SLOTS[2][1] + ROSTER_SLOTS[3][1]
                ))
            else:
                have = pos_counts.get(slot, 0)
            res.append((slot, need, have))
        return res

    # Total picks required per team
    total_required = sum(count for _, count in ROSTER_SLOTS)
    rounds = total_required

    precomputed_top = load_top_players_csv()

    # LLM call logging
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "draft_results")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"llm_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    def log_llm_call(team_name, call_type, payload, response):
        record = {
            "time": datetime.now().isoformat(),
            "team": team_name,
            "call_type": call_type,
            "input": payload,
            "output": response,
        }
        with open(log_path, "a") as lf:
            lf.write(json.dumps(record) + "\n")

    for rnd in range(rounds):
        for team in teams:
            # Get top remaining per position (hidden rank, but shuffled in helper)
            pos_to = top_remaining_by_position(hidden_ranked, drafted, top_k=10)
            # Filter out players that don't fit remaining roster slots
            valid_options = []
            for pos, plist in pos_to.items():
                for p in plist:
                    if p["Name"] not in drafted and can_add(team["picks"], p["Name"]):
                        valid_options.append(p)
            # Deduplicate by name
            seen = set()
            valid_options = [p for p in valid_options if not (p["Name"] in seen or seen.add(p["Name"]))]

            # Prepare first LLM call: ask for 5 players to research
            state = roster_state_string(roster_slots_filled(team["picks"]))
            # Prepare per-position top 5 suggestions from CSV (available + roster-fit) and live fallbacks
            suggestions = {}
            for pos in ("QB", "RB", "WR", "TE"):
                # Do not use live fallback names to avoid any abbreviated names slipping in
                live_fallback = []
                suggestions[pos] = top5_available_from_csv(
                    precomputed_top,
                    pos,
                    drafted,
                    lambda nm: can_add(team["picks"], nm),
                    live_fallback,
                )
            suggested_lines = [f"Suggested Top {pos}: " + ", ".join(suggestions[pos]) for pos in ("QB","RB","WR","TE")]
            context_lines = suggested_lines
            context = "\n".join(context_lines)
            team_info = (
                "Fantasy Football Draft Setup:\n"
                f"{format_roster_format()}\n"
                f"{format_scoring_format()}\n\n"
                "You are drafting in a fantasy football snake draft. By the end of the draft, you MUST draft a player for each position on your roster. If you have an empty roster position, you MUST draft a player for that position.\n"
                f"Current roster needs: {state}.\n"
                "Below are the suggested top-5 AVAILABLE players by position (not yet drafted).\n"
                f"{context}\n\n"
                "Return EXACTLY 5 player names you want to research next, comma-separated, using the names EXACTLY as shown above."
            )

            # First call: request 5 candidate names (STRICT JSON)
            first_prompt = (
                team_info + "\n\n" +
                "Respond ONLY with a JSON object of the form {\"candidates\": [\"name1\", \"name2\", \"name3\", \"name4\", \"name5\"]}. "
                "Choose EXACTLY 5 names from the suggested lists above. Do not include any extra text."
            )
            first_resp_raw = llm.call_with_prompt(team["model"]["arn"], first_prompt)
            log_llm_call(team["name"], "request_candidates", {"prompt": first_prompt}, first_resp_raw)

            requested_names = []
            # Try strict JSON parse
            if isinstance(first_resp_raw, str):
                txt = first_resp_raw.strip()
                if "{" in txt and "}" in txt:
                    try:
                        js = json.loads(txt[txt.find("{"): txt.rfind("}")+1])
                        if isinstance(js, dict) and isinstance(js.get("candidates"), list):
                            requested_names = [str(x).strip() for x in js["candidates"]][:5]
                    except Exception:
                        pass
            elif isinstance(first_resp_raw, dict) and isinstance(first_resp_raw.get("candidates"), list):
                requested_names = [str(x).strip() for x in first_resp_raw["candidates"]][:5]
            # Fallback naive split
            if not requested_names and isinstance(first_resp_raw, str):
                parts = [p.strip() for p in first_resp_raw.replace("\n", ",").split(",") if p.strip()]
                requested_names = parts[:5]
            if not requested_names:
                fallback = []
                for pos in ("QB","RB","WR","TE"):
                    fallback.extend(suggestions[pos])
                requested_names = fallback[:5]

            # Canonicalize names to suggested tokens
            def canonicalize_name(nm):
                key = ''.join(ch for ch in nm.lower() if ch.isalnum())
                candidates = set()
                for pos in ("QB","RB","WR","TE"):
                    candidates.update(suggestions[pos])
                for c in candidates:
                    ckey = ''.join(ch for ch in c.lower() if ch.isalnum())
                    if key == ckey or key in ckey or ckey in key:
                        return c
                return nm
            requested_names = [canonicalize_name(n) for n in requested_names]

            # Gather summaries strictly for these 5 names
            candidates = []
            for nm in requested_names:
                if nm in drafted:
                    continue
                pool_entry = next((p for p in player_pool if p["Name"] == nm), None)
                if not pool_entry:
                    continue
                candidates.append((nm, format_player_one_liner(dh, nm)))
                if len(candidates) == 5:
                    break

            # Second call: ask to choose one
            cand_lines = "\n".join([f"- {nm}: {summary}" for nm, summary in candidates])
            allowed_names_list = [nm for nm, _ in candidates]
            allowed_names = ", ".join(allowed_names_list)
            second_prompt = (
                "You are drafting. Here are brief summaries for exactly 5 candidates.\n"
                f"{cand_lines}\n\n"
                f"Respond ONLY with a JSON object of the form {{\"pick\": \"<one name from: {allowed_names}>\"}}."
            )
            second_resp_raw = llm.call_with_prompt(team["model"]["arn"], second_prompt)
            log_llm_call(team["name"], "final_pick", {"prompt": second_prompt}, second_resp_raw)
            pick_name = ""
            if isinstance(second_resp_raw, str):
                txt2 = second_resp_raw.strip()
                if "{" in txt2 and "}" in txt2:
                    try:
                        js2 = json.loads(txt2[txt2.find("{"): txt2.rfind("}")+1])
                        if isinstance(js2, dict) and isinstance(js2.get("pick"), str):
                            pick_name = js2["pick"].strip()
                    except Exception:
                        pass
                if not pick_name:
                    pick_name = txt2
            elif isinstance(second_resp_raw, dict) and isinstance(second_resp_raw.get("pick"), str):
                pick_name = second_resp_raw["pick"].strip()

            # Validate pick; fallback if invalid
            valid_names = {nm for nm, _ in candidates}
            if pick_name not in valid_names:
                # fallback to best available that fits
                for p in hidden_ranked:
                    if p["Name"] not in drafted and can_add(team["picks"], p["Name"]):
                        pick_name = p["Name"]
                        break

            # Commit pick
            drafted.add(pick_name)
            pick_player = next(pp for pp in player_pool if pp["Name"] == pick_name)
            team["picks"].append(pick_player)
            print(f"Round {rnd+1} - {team['name']} drafted {pick_name} ({pick_player['Position']})")

    # Save to CSV per team
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "draft_results")
    os.makedirs(out_dir, exist_ok=True)
    for team in teams:
        # Include model nickname in filename
        safe_name = team["name"].replace(" ", "_")
        filename = f"{safe_name}_2024.csv"
        path = os.path.join(out_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Team", "Position", "FantasyPoints", "Model", "ModelName"])
            for p in team["picks"]:
                writer.writerow([p["Name"], p["Team"], p["Position"], f"{p['FantasyPoints']:.2f}", team["model"]["arn"], team["model"]["name"]])
        print(f"Saved roster to {path}")


if __name__ == "__main__":
    draft_simulation()


