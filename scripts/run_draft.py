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
from ffbench.config import get_scoring, get_roster_slots, get_models, get_num_teams, format_roster_format, format_scoring_format, get_season_weeks


ROSTER_SLOTS = get_roster_slots()

SCORING = get_scoring()
def _strip_code_fences(text: str):
    s = text.strip()
    if s.startswith("```"):
        # remove first fence line
        s = s.split("\n", 1)[-1]
        # remove trailing fence if present
        if s.endswith("```"):
            s = s.rsplit("\n", 1)[0]
    return s.strip()

def _parse_pick_from_text(text: str) -> str:
    import re
    text = _strip_code_fences(text)
    # Try to find "pick": "name"
    match = re.search(r'"pick"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Also try without quotes around value
    match = re.search(r'"pick"\s*:\s*([^,\n}]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('"').strip("'")
    # Fallback: look for any name that matches candidates (but we'll handle that later)
    return ""

def _parse_candidates_from_text(text: str) -> list:
    import re
    text = _strip_code_fences(text)
    # Try to find "candidates": [ ... ]
    match = re.search(r'"candidates"\s*:\s*\[([^\]]*)\]', text, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1)
        names = re.findall(r'"([^"]+)"', content)
        return [n.strip() for n in names if n.strip()]
    # Fallback: split by comma
    parts = [p.strip().strip('"').strip("'") for p in text.replace("\n", ",").split(",") if p.strip()]
    return parts

def _extract_json(text: str):
    import json as _json
    s = _strip_code_fences(text)
    # try direct JSON
    try:
        return _json.loads(s)
    except Exception:
        pass
    # try to find first {...} or [ ... ]
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return _json.loads(s[start:end+1])
        except Exception:
            pass
    start = s.find('[')
    end = s.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            return _json.loads(s[start:end+1])
        except Exception:
            pass
    return None



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
    season_weeks = get_season_weeks()
    for wk in range(1, season_weeks + 1):
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
    sim_id = os.environ.get("FFBENCH_SIM_ID")
    if not sim_id:
        latest_path = os.path.join(root, "data", "simulations", "latest_simulation_id.txt")
        if os.path.exists(latest_path):
            with open(latest_path) as f:
                sim_id = f.read().strip()
    search_paths = []
    if sim_id:
        search_paths.append(os.path.join(root, "data", "simulations", sim_id, "draft_results", "TopPlayers_2024_Projections_PPR.csv"))
    search_paths.append(os.path.join(root, "data", "draft_results", "TopPlayers_2024_Projections_PPR.csv"))
    path = next((p for p in search_paths if os.path.exists(p)), None)
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
    season_weeks = get_season_weeks()
    for week in range(1, season_weeks + 1):
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

    # Load models list and randomize draft order for snake draft
    models = get_models()
    # Create team indices and randomize them for the initial draft order
    team_indices = list(range(len(models)))
    random.shuffle(team_indices)

    # Build teams based on randomized model order; include nickname in team name
    teams = [
        {"name": f"{models[idx]['name']}_Team_{i+1}", "model": models[idx], "picks": [], "original_idx": idx, "draft_position": i+1}
        for i, idx in enumerate(team_indices)
    ]

    # Roster constraints
    base_slots = [(slot, count) for slot, count in ROSTER_SLOTS]

    llm = LLMManager()
    drafted = set()  # Current round drafted players
    drafted_players = set()  # Global set of all drafted players

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

    # Display draft order
    print("Snake Draft Order:")
    for i, team in enumerate(teams):
        print(f"  Pick {i+1}: {team['name']}")
    print()

    precomputed_top = load_top_players_csv()

    # LLM call logging under simulation folder
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_id_env = os.environ.get("FFBENCH_SIM_ID")
    if not sim_id_env:
        latest_path = os.path.join(root, "data", "simulations", "latest_simulation_id.txt")
        if os.path.exists(latest_path):
            with open(latest_path) as f:
                sim_id_env = f.read().strip()
        else:
            sim_id_env = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sim_root = os.path.join(root, "data", "simulations", sim_id_env)
    log_dir = os.path.join(sim_root, "draft_results")
    os.makedirs(log_dir, exist_ok=True)

    # Check if there's an existing log file to resume from
    existing_log_path = None
    import glob
    log_pattern = os.path.join(log_dir, "llm_calls_*.jsonl")
    existing_logs = glob.glob(log_pattern)
    if existing_logs:
        existing_log_path = existing_logs[0]  # Use the first (should be only one)
        log_path = existing_log_path  # Append to existing log
        print(f"Resuming from existing log: {existing_log_path}")
    else:
        log_path = os.path.join(log_dir, f"llm_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        print(f"Starting new log: {log_path}")

    # Parse existing log to resume draft state if it exists
    drafted_players = set()
    team_rosters = {team["name"]: [] for team in teams}
    resume_from_round = 0
    resume_from_team_idx = 0

    # Helper to extract a pick string from a logged output (dict or string)
    def _extract_pick_from_output(output_val: object) -> str:
        pick_val = ""
        if isinstance(output_val, dict):
            pv = output_val.get("pick")
            if isinstance(pv, str):
                pick_val = pv.strip()
        elif isinstance(output_val, str):
            js = _extract_json(output_val)
            if isinstance(js, dict) and isinstance(js.get("pick"), str):
                pick_val = js["pick"].strip()
            else:
                pick_val = _parse_pick_from_text(output_val).strip()
        return pick_val

    if existing_log_path:
        print("Parsing existing log to resume draft state...")
        # Build a fast lookup for player existence
        player_pool_names = {p["Name"] for p in player_pool}
        with open(existing_log_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("call_type") == "final_pick":
                        output = entry.get("output")
                        team_name = entry.get("team")
                        player_name = _extract_pick_from_output(output)
                        # Only count as drafted if we have a valid player name and it's in our player pool
                        if player_name and player_name in player_pool_names:
                            if player_name not in drafted_players:
                                drafted_players.add(player_name)
                                if team_name in team_rosters:
                                    # Avoid duplicate append to roster for same player
                                    if not any(p["Name"] == player_name for p in team_rosters[team_name]):
                                        player = next((p for p in player_pool if p["Name"] == player_name), None)
                                        if player:
                                            team_rosters[team_name].append(player)
                except Exception:
                    continue

        print(f"Found {len(drafted_players)} already drafted players")
        if drafted_players:
            print(f"Drafted players: {sorted(list(drafted_players))[:10]}{'...' if len(drafted_players) > 10 else ''}")

        # Debug: Show team rosters after resume
        for team_name, roster in team_rosters.items():
            print(f"Team {team_name}: {len(roster)} players")

        # Determine where to resume - find the last activity and check if it was completed
        last_team = None
        last_call_type = None
        last_pick_completed = False
        total_successful_picks = len(drafted_players)

        with open(existing_log_path) as f:
            lines = f.readlines()

        # Process entries to find the true last state
        for i, line in enumerate(lines):
            try:
                entry = json.loads(line.strip())
                team_name = entry["team"]
                call_type = entry["call_type"]

                if call_type == "final_pick":
                    # Check if this final_pick has a valid pick (dict or string output)
                    output = entry.get("output")
                    player_name = _extract_pick_from_output(output)
                    if player_name and player_name in [p["Name"] for p in player_pool]:
                        last_pick_completed = True
                        last_team = team_name
                        last_call_type = call_type
                    else:
                        # This final_pick didn't complete successfully
                        last_pick_completed = False
                        last_team = team_name
                        last_call_type = call_type
                elif call_type == "request_candidates":
                    last_team = team_name
                    last_call_type = call_type
                    last_pick_completed = False
            except:
                continue

        # Determine resume point based on last state and total successful picks
        if last_call_type == "request_candidates":
            # Need to complete this team's pick (they requested candidates but didn't finish)
            resume_from_team_idx = next(i for i, t in enumerate(teams) if t["name"] == last_team)
            resume_from_round = total_successful_picks // len(teams)
            print(f"Resuming from round {resume_from_round + 1}, team {last_team} (incomplete pick)")
        elif last_call_type == "final_pick" and not last_pick_completed:
            # Final pick was attempted but didn't complete - retry this team's pick
            resume_from_team_idx = next(i for i, t in enumerate(teams) if t["name"] == last_team)
            resume_from_round = total_successful_picks // len(teams)
            print(f"Resuming from round {resume_from_round + 1}, team {last_team} (retrying incomplete pick)")
        else:
            # Last pick was completed successfully - move to next team
            current_team_idx = next(i for i, t in enumerate(teams) if t["name"] == last_team)
            resume_from_team_idx = (current_team_idx + 1) % len(teams)
            if resume_from_team_idx == 0:
                resume_from_round = total_successful_picks // len(teams) + 1
            else:
                resume_from_round = total_successful_picks // len(teams)
            print(f"Resuming from round {resume_from_round + 1}, next team after {last_team}")

        # Ensure global drafted list is reflected in current round's drafted set
        drafted.update(drafted_players)

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

    for rnd in range(resume_from_round, rounds):
        # Snake draft: reverse order for odd rounds (1-indexed)
        draft_order = teams if rnd % 2 == 0 else list(reversed(teams))

        # Ensure drafted set includes all previously drafted players
        drafted.update(drafted_players)

        # Determine starting team index for this round
        start_team_idx = resume_from_team_idx if rnd == resume_from_round else 0

        for team_idx in range(start_team_idx, len(draft_order)):
            team = draft_order[team_idx]

            # Reset resume variables after first iteration
            if rnd == resume_from_round and team_idx == start_team_idx:
                resume_from_round = 0
                resume_from_team_idx = 0

            # Combine existing roster with current picks for accurate state
            current_roster = team_rosters[team["name"]] + team["picks"]

            # Get top remaining per position (hidden rank, but shuffled in helper)
            pos_to = top_remaining_by_position(hidden_ranked, drafted, top_k=10)
            # Filter out players that don't fit remaining roster slots
            valid_options = []
            for pos, plist in pos_to.items():
                for p in plist:
                    if p["Name"] not in drafted and can_add(current_roster, p["Name"]):
                        valid_options.append(p)
            # Deduplicate by name
            seen = set()
            valid_options = [p for p in valid_options if not (p["Name"] in seen or seen.add(p["Name"]))]

            # Prepare first LLM call: ask for 5 players to research
            state = roster_state_string(roster_slots_filled(current_roster))
            # Prepare per-position top 5 suggestions from CSV (available + roster-fit) and live fallbacks
            suggestions = {}
            for pos in ("QB", "RB", "WR", "TE"):
                # Do not use live fallback names to avoid any abbreviated names slipping in
                live_fallback = []
                suggestions[pos] = top5_available_from_csv(
                    precomputed_top,
                    pos,
                    drafted,
                    lambda nm: can_add(current_roster, nm),
                    live_fallback,
                )
            suggested_lines = [f"Suggested Top {pos}: " + ", ".join(suggestions[pos]) for pos in ("QB","RB","WR","TE")]
            # Format current roster list with positions
            if current_roster:
                roster_list = "\n" + "\n".join([f"- {p['Name']} ({p['Position']})" for p in current_roster])
            else:
                roster_list = "\n- None yet"
            context_lines = suggested_lines
            context = "\n".join(context_lines)
            team_info = (
                "Fantasy Football Draft Setup:\n"
                f"{format_roster_format()}\n"
                f"{format_scoring_format()}\n\n"
                "You are drafting in a fantasy football snake draft. By the end of the draft, you MUST draft a player for each position on your roster. If you have an empty roster position, you MUST draft a player for that position.\n"
                f"Current roster needs: {state}.\n"
                "Your current roster so far (Name - Position):" + roster_list + "\n"
                "Below are the suggested top-5 AVAILABLE players by position sorted by projected points fro the upcoming year (not yet drafted).\n"
                f"{context}\n\n"
                "Return EXACTLY 5 player names you want to research next, comma-separated, using the names EXACTLY as shown above."
            )

            # First call: request 5 candidate names (STRICT JSON, but parser accepts text or JSON)
            first_prompt = team_info + "\n\n" + (
                "IMPORTANT: You are ONLY allowed to use the data explicitly provided in this prompt. "
                "You cannot perform web searches, access external databases, or use any additional information not provided here. "
                "All decisions must be based solely on the player names and context given above.\n\n"
                "Respond ONLY with a JSON object of the form {\"candidates\": [\"name1\", \"name2\", \"name3\", \"name4\", \"name5\"]}. "
                "Choose EXACTLY 5 names from the suggested lists above. Do not include any extra text."
            )
            first_resp_raw = llm.call_with_prompt(team["model"]["id"], first_prompt, response_format={"type":"json_object"})
            log_llm_call(team["name"], "request_candidates", {"prompt": first_prompt}, first_resp_raw)

            requested_names = []
            # Try strict JSON parse
            if isinstance(first_resp_raw, dict) and isinstance(first_resp_raw.get("candidates"), list):
                requested_names = [str(x).strip() for x in first_resp_raw["candidates"]][:5]
            elif isinstance(first_resp_raw, str):
                js = _extract_json(first_resp_raw)
                if isinstance(js, dict) and isinstance(js.get("candidates"), list):
                    requested_names = [str(x).strip() for x in js["candidates"]][:5]
            # Fallback parse from text
            if not requested_names and isinstance(first_resp_raw, str):
                parsed = _parse_candidates_from_text(first_resp_raw)
                requested_names = parsed[:5]
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
            second_rule = (f"Respond ONLY with a JSON object of the form {{\"pick\": \"<one name from: {allowed_names}>\"}}.")
            second_prompt = (
                "You are drafting.\n"
                "Your current roster so far (Name - Position):" + ("\n" + "\n".join([f"- {p['Name']} ({p['Position']})" for p in current_roster]) if current_roster else "\n- None yet") + "\n\n"
                "Here are brief summaries for exactly 5 candidates.\n"
                f"{cand_lines}\n\n"
                "IMPORTANT: You are ONLY allowed to use the data explicitly provided in this prompt. "
                "You cannot perform web searches, access external databases, or use any additional information not provided here. "
                "All decisions must be based solely on the player summaries given above.\n\n"
                f"{second_rule}"
            )
            second_resp_raw = llm.call_with_prompt(team["model"]["id"], second_prompt, response_format={"type":"json_object"})
            log_llm_call(team["name"], "final_pick", {"prompt": second_prompt}, second_resp_raw)
            pick_name = ""
            if isinstance(second_resp_raw, dict) and isinstance(second_resp_raw.get("pick"), str):
                pick_name = second_resp_raw["pick"].strip()
            elif isinstance(second_resp_raw, str):
                js2 = _extract_json(second_resp_raw)
                if isinstance(js2, dict) and isinstance(js2.get("pick"), str):
                    pick_name = js2["pick"].strip()
                else:
                    # fallback to parse from text
                    pick_name = _parse_pick_from_text(second_resp_raw)

            # Normalize pick to one of the candidate tokens if possible
            if pick_name:
                def _canon(s: str) -> str:
                    return ''.join(ch for ch in s.lower() if ch.isalnum())
                cand_map = { _canon(nm): nm for nm, _ in candidates }
                pn = _canon(pick_name)
                if pn in cand_map:
                    pick_name = cand_map[pn]

            # Validate pick; fallback if invalid
            valid_names = {nm for nm, _ in candidates}
            if pick_name not in valid_names:
                # Try to accept any exact match in player pool if available and fits roster
                pool_match = next((p for p in player_pool if p["Name"].lower() == pick_name.lower()), None)
                if pool_match and pool_match["Name"] not in drafted and can_add(current_roster, pool_match["Name"]):
                    pick_name = pool_match["Name"]
                else:
                    # fallback to best available that fits
                    for p in hidden_ranked:
                        if p["Name"] not in drafted and p["Name"] not in drafted_players and can_add(current_roster, p["Name"]):
                            pick_name = p["Name"]
                            break

            # Commit pick
            drafted.add(pick_name)
            drafted_players.add(pick_name)  # Also update the global set
            pick_player = next(pp for pp in player_pool if pp["Name"] == pick_name)
            team["picks"].append(pick_player)
            # Calculate overall pick number in snake draft
            pick_num = rnd * len(teams) + draft_order.index(team) + 1
            print(f"Round {rnd+1}, Pick {pick_num} - {team['name']} drafted {pick_name} ({pick_player['Position']})")

    # Save to CSV per team under simulation folder
    out_dir = os.path.join(sim_root, "draft_results")
    os.makedirs(out_dir, exist_ok=True)
    for team in teams:
        # Include model nickname in filename
        safe_name = team["name"].replace(" ", "_")
        filename = f"{safe_name}_2024.csv"
        path = os.path.join(out_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Team", "Position", "FantasyPoints", "Model", "ModelName"])
            # Include both existing roster and new picks
            for p in team_rosters[team["name"]] + team["picks"]:
                writer.writerow([p["Name"], p["Team"], p["Position"], f"{p['FantasyPoints']:.2f}", team["model"]["id"], team["model"]["name"]])
        print(f"Saved roster to {path}")


def resume_draft(simulation_id=None):
    """Resume an interrupted draft from the simulation folder."""
    import json

    # Determine simulation folder
    if not simulation_id:
        # Try to find latest simulation
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sim_meta_dir = os.path.join(root, "data", "simulations")
        latest_file = os.path.join(sim_meta_dir, "latest_simulation_id.txt")
        if os.path.exists(latest_file):
            with open(latest_file) as f:
                simulation_id = f.read().strip()
        else:
            print("ERROR: No simulation_id provided and no latest_simulation_id.txt found")
            return

    sim_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "simulations", simulation_id)
    draft_dir = os.path.join(sim_root, "draft_results")
    llm_log_path = os.path.join(draft_dir, "llm_calls_*.jsonl")

    # Find the LLM calls log file
    import glob
    log_files = glob.glob(llm_log_path)
    if not log_files:
        print(f"ERROR: No LLM calls log found in {draft_dir}")
        return

    log_file = log_files[0]  # Use the first (should be only one)

    print(f"Resuming draft from: {simulation_id}")
    print(f"Using log file: {log_file}")

    # Parse the log to reconstruct draft state
    drafted_players = set()
    team_picks = {}  # team_name -> list of players
    last_team = None
    last_call_type = None

    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                team_name = entry["team"]
                call_type = entry["call_type"]

                if team_name not in team_picks:
                    team_picks[team_name] = []

                # Track the last activity
                last_team = team_name
                last_call_type = call_type

                # If this was a successful final_pick, add the player
                if call_type == "final_pick":
                    output = entry.get("output", {})
                    if isinstance(output, dict) and "pick" in output:
                        player_name = output["pick"]
                        if player_name not in drafted_players:
                            drafted_players.add(player_name)
                            team_picks[team_name].append({"Name": player_name})
                            print(f"✓ {team_name} drafted {player_name}")

            except json.JSONDecodeError:
                continue

    print(f"\nDraft state reconstructed:")
    print(f"- Drafted players: {len(drafted_players)}")
    for team, picks in team_picks.items():
        print(f"- {team}: {len(picks)} players")

    # Now resume the draft from the last state
    # This is a simplified version - in practice you'd want to restart the full draft
    # with the current state, but for now let's just run a fresh draft
    print(f"\nLast activity: {last_team} - {last_call_type}")
    print("Note: For a complete resume, the draft script would need to be modified to accept initial state.")
    print("For now, you can restart the draft manually or modify the script to handle resumption.")


def complete_draft_quickly(simulation_id=None):
    """Complete an interrupted draft by assigning remaining players to teams without LLM calls."""
    import json
    import random

    # Determine simulation folder
    if not simulation_id:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sim_meta_dir = os.path.join(root, "data", "simulations")
        latest_file = os.path.join(sim_meta_dir, "latest_simulation_id.txt")
        if os.path.exists(latest_file):
            with open(latest_file) as f:
                simulation_id = f.read().strip()
        else:
            print("ERROR: No simulation_id found")
            return

    print(f"Completing draft for: {simulation_id}")

    # Load existing data
    dh = DataHandler()
    player_pool = build_player_pool_2024(dh)
    models = get_models()

    # Create teams
    team_indices = list(range(len(models)))
    random.shuffle(team_indices)
    teams = [
        {"name": f"{models[idx]['name']}_Team_{i+1}", "model": models[idx], "picks": []}
        for i, idx in enumerate(team_indices)
    ]

    # Load existing LLM calls to see what was already drafted
    sim_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "simulations", simulation_id)
    draft_dir = os.path.join(sim_root, "draft_results")

    drafted_players = set()
    team_rosters = {team["name"]: [] for team in teams}

    # Find and parse LLM logs
    import glob
    log_pattern = os.path.join(draft_dir, "llm_calls_*.jsonl")
    log_files = glob.glob(log_pattern)

    for log_file in log_files:
        with open(log_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("call_type") == "final_pick":
                        output = entry.get("output", {})
                        if isinstance(output, dict) and "pick" in output:
                            player_name = output["pick"]
                            team_name = entry["team"]
                            drafted_players.add(player_name)
                            if team_name in team_rosters:
                                # Find the player in pool
                                player = next((p for p in player_pool if p["Name"] == player_name), None)
                                if player:
                                    team_rosters[team_name].append(player)
                except:
                    continue

    print(f"Found {len(drafted_players)} already drafted players")

    # Assign remaining players to teams that need them
    roster_slots = get_roster_slots()
    total_slots = sum(count for _, count in roster_slots)

    remaining_players = [p for p in player_pool if p["Name"] not in drafted_players]

    for team in teams:
        current_count = len(team_rosters[team["name"]])
        needed = total_slots - current_count

        if needed > 0:
            # Take next available players
            team_picks = remaining_players[:needed]
            team_rosters[team["name"]].extend(team_picks)
            remaining_players = remaining_players[needed:]

            print(f"Assigned {len(team_picks)} players to {team['name']}")

    # Save completed rosters
    out_dir = os.path.join(sim_root, "draft_results")
    os.makedirs(out_dir, exist_ok=True)

    for team in teams:
        safe_name = team["name"].replace(" ", "_")
        filename = f"{safe_name}_2024.csv"
        path = os.path.join(out_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Team", "Position", "FantasyPoints", "Model", "ModelName"])
            for p in team_rosters[team["name"]]:
                writer.writerow([p["Name"], p["Team"], p["Position"], f"{p['FantasyPoints']:.2f}", team["model"]["id"], team["model"]["name"]])
        print(f"Saved roster to {path}")

    print(f"\nDraft completed! All {len(teams)} teams now have full rosters.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        simulation_id = sys.argv[2] if len(sys.argv) > 2 else None
        resume_draft(simulation_id)
    elif len(sys.argv) > 1 and sys.argv[1] == "--complete":
        simulation_id = sys.argv[2] if len(sys.argv) > 2 else None
        complete_draft_quickly(simulation_id)
    else:
        draft_simulation()


