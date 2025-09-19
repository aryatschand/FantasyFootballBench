#!/usr/bin/env python3
import os
import sys
import csv
import json
from datetime import datetime
import random
from collections import defaultdict, Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffbench.data_handler import DataHandler
from ffbench.scoring import calculate_fantasy_points
from ffbench.llm import LLMManager
from ffbench.config import get_scoring, get_roster_slots, get_models, format_roster_format, format_scoring_format


SCORING = get_scoring()
def _strip_code_fences(text: str):
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("\n", 1)[0]
    return s.strip()

def _extract_json(text: str):
    import json as _json
    s = _strip_code_fences(text)
    try:
        return _json.loads(s)
    except Exception:
        pass
    start = s.find('{'); end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return _json.loads(s[start:end+1])
        except Exception:
            pass
    start = s.find('['); end = s.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            return _json.loads(s[start:end+1])
        except Exception:
            pass
    return None

def _parse_trade_proposal(text: str):
    import re
    text = _strip_code_fences(text)
    # Extract target_team
    target_match = re.search(r'"target_team"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    target_team = target_match.group(1).strip() if target_match else None
    # Extract give: list of names
    give_match = re.search(r'"give"\s*:\s*\[([^\]]*)\]', text, re.IGNORECASE | re.DOTALL)
    give_names = []
    if give_match:
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', give_match.group(1), re.IGNORECASE)
        give_names = [n.strip() for n in names]
    # Extract receive
    receive_match = re.search(r'"receive"\s*:\s*\[([^\]]*)\]', text, re.IGNORECASE | re.DOTALL)
    receive_names = []
    if receive_match:
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', receive_match.group(1), re.IGNORECASE)
        receive_names = [n.strip() for n in names]
    # Extract rationale
    rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', text, re.IGNORECASE | re.DOTALL)
    rationale = rationale_match.group(1).strip() if rationale_match else ""
    return {"target_team": target_team, "give": give_names, "receive": receive_names, "rationale": rationale}

def _parse_trade_decision(text: str):
    import re
    text = _strip_code_fences(text)
    # Extract decision
    decision_match = re.search(r'"decision"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    decision = decision_match.group(1).strip().lower() if decision_match else "counter"
    result = {"decision": decision}
    if decision == "counter":
        # Extract give and receive like above
        give_match = re.search(r'"give"\s*:\s*\[([^\]]*)\]', text, re.IGNORECASE | re.DOTALL)
        give_names = []
        if give_match:
            names = re.findall(r'"name"\s*:\s*"([^"]+)"', give_match.group(1), re.IGNORECASE)
            give_names = [n.strip() for n in names]
        receive_match = re.search(r'"receive"\s*:\s*\[([^\]]*)\]', text, re.IGNORECASE | re.DOTALL)
        receive_names = []
        if receive_match:
            names = re.findall(r'"name"\s*:\s*"([^"]+)"', receive_match.group(1), re.IGNORECASE)
            receive_names = [n.strip() for n in names]
        rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', text, re.IGNORECASE | re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else ""
        result.update({"give": give_names, "receive": receive_names, "rationale": rationale})
    return result



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


def load_teams_from_draft(draft_dir):
    teams = []
    for fname in sorted(os.listdir(draft_dir)):
        if not fname.endswith("_2024.csv"):
            continue
        path = os.path.join(draft_dir, fname)
        # Extract team name from filename (handle new format with model ARN)
        base_name = os.path.splitext(os.path.basename(fname))[0]
        # Remove _2024 suffix and extract team name before model ARN
        if "_us.anthropic.claude-" in base_name:
            name = base_name.split("_us.anthropic.claude-")[0]
        else:
            name = base_name.replace("_2024", "")

        roster = []
        model_arn = None
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                roster.append({
                    "Name": row.get("Name"),
                    "Team": row.get("Team"),
                    "Position": row.get("Position"),
                    "FantasyPoints": float(row.get("FantasyPoints", 0) or 0),
                })
                # Extract model ARN from the first row
                if model_arn is None and row.get("Model"):
                    model_arn = row.get("Model")

        teams.append({"name": name, "roster": roster, "model_id": model_arn or "unknown", "csv_path": path})
    return teams


def select_starting_lineup(roster):
    # Use roster slots from config
    roster_slots = get_roster_slots()
    starters = []
    bench = []
    pos_buckets = defaultdict(list)
    for p in roster:
        pos_buckets[p["Position"]].append(p)

    # Helper to pop first available from a position
    def pop_first(position):
        if pos_buckets[position]:
            starters.append(pos_buckets[position].pop(0))

    # Assign slots by config order; handle FLEX specially
    for slot, count in roster_slots:
        if slot == "FLEX":
            for _ in range(count):
                for pos in ("RB", "WR", "TE"):
                    if pos_buckets[pos]:
                        starters.append(pos_buckets[pos].pop(0))
                        break
        elif slot == "BENCH":
            # handled after
            continue
        else:
            for _ in range(count):
                pop_first(slot)

    # Remaining go to bench
    for pos in pos_buckets:
        bench.extend(pos_buckets[pos])

    return starters, bench


# Removed parse_abbrev and resolve_full_name_2024 functions - now using full names directly


def player_week_points(dh, name_full, week):
    df = dh.read_player_game_stats_by_week(2024, week)
    if df.empty:
        return 0.0
    row = df[df["Name"] == name_full]
    if row.empty:
        return 0.0
    stats_map = normalize_stat_keys(row.iloc[0])
    return float(calculate_fantasy_points(stats_map, SCORING))


def validate_trade_proposal(proposer_team, receiver_team, give_names, receive_names):
    # Equal non-zero counts, max 3 per side
    if not isinstance(give_names, list) or not isinstance(receive_names, list):
        return False, "Invalid list types"
    if len(give_names) == 0 or len(receive_names) == 0:
        return False, "Empty sides"
    if len(give_names) != len(receive_names):
        return False, "Unequal counts"
    if len(give_names) > 3:
        return False, "Too many players per side"
    prop_names = {p["Name"] for p in proposer_team["roster"]}
    recv_names = {p["Name"] for p in receiver_team["roster"]}
    if any(n not in prop_names for n in give_names):
        return False, "Proposer does not own all give players"
    if any(n not in recv_names for n in receive_names):
        return False, "Receiver does not own all receive players"
    return True, "ok"


def apply_trade(proposer_team, receiver_team, give_names, receive_names):
    # Remove and add players by full name
    give_set = set(give_names); recv_set = set(receive_names)
    new_prop = [p for p in proposer_team["roster"] if p["Name"] not in give_set]
    new_recv = [p for p in receiver_team["roster"] if p["Name"] not in recv_set]
    # Add received players
    add_to_prop = [p for p in receiver_team["roster"] if p["Name"] in recv_set]
    add_to_recv = [p for p in proposer_team["roster"] if p["Name"] in give_set]
    new_prop.extend(add_to_prop)
    new_recv.extend(add_to_recv)
    proposer_team["roster"] = new_prop
    receiver_team["roster"] = new_recv


def write_team_csv(team):
    # Persist team roster back to its CSV path; keep Model column if known
    path = team.get("csv_path")
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        headers = ["Name", "Team", "Position", "FantasyPoints", "Model"]
        writer.writerow(headers)
        for p in team["roster"]:
            writer.writerow([p.get("Name"), p.get("Team"), p.get("Position"), f"{float(p.get('FantasyPoints', 0) or 0):.2f}", team.get("model_id", "")])


def simulate_season():
    random.seed(42)
    dh = DataHandler()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_id = os.environ.get("FFBENCH_SIM_ID")
    if not sim_id:
        latest_path = os.path.join(root, "data", "simulations", "latest_simulation_id.txt")
        if os.path.exists(latest_path):
            with open(latest_path) as f:
                sim_id = f.read().strip()
    if not sim_id:
        sim_id = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sim_root = os.path.join(root, "data", "simulations", sim_id)
    draft_dir = os.path.join(sim_root, "draft_results")
    out_dir = os.path.join(sim_root, "season_results")
    os.makedirs(out_dir, exist_ok=True)

    teams = load_teams_from_draft(draft_dir)
    # Attach models to teams; keep team names as created during draft
    models = get_models()
    for i, t in enumerate(teams):
        model = models[i % len(models)]
        t["model_id"] = model["id"]
        t["model_name"] = model["name"]
    # Initial default lineups (used if LLM fails)
    for t in teams:
        starters, bench = select_starting_lineup(t["roster"])
        t["default_starters"] = starters
        t["default_bench"] = bench

    # Name resolution no longer needed - using full names directly

    # Initialize standings
    standings = {
        t["name"]: {"wins": 0, "losses": 0, "ties": 0, "points_for": 0.0}
        for t in teams
    }

    llm = LLMManager()

    def summarize_weekly_year(dh_local, full_name, year):
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
        found_any = False
        for wk in range(1, 18):
            dfw = dh_local.read_player_game_stats_by_week(year, wk)
            if dfw.empty:
                continue
            row = dfw[dfw["Name"] == full_name]
            if row.empty:
                continue
            found_any = True
            r = row.iloc[0]
            for k in total.keys():
                total[k] += float(r.get(k, 0) or 0)
        return found_any, total

    def summarize_2024_to_week(full_name, upto_week):
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
        for wk in range(1, upto_week + 1):
            df = dh.read_player_game_stats_by_week(2024, wk)
            if df.empty:
                continue
            row = df[df["Name"] == full_name]
            if row.empty:
                continue
            r = row.iloc[0]
            for k in total.keys():
                total[k] += float(r.get(k, 0) or 0)
        return total

    def last_3_weeks(full_name, current_week):
        weeks = []
        for wk in range(max(1, current_week - 3), current_week):
            df = dh.read_player_game_stats_by_week(2024, wk)
            if df.empty:
                weeks.append((wk, {}))
                continue
            row = df[df["Name"] == full_name]
            if row.empty:
                weeks.append((wk, {}))
                continue
            r = row.iloc[0]
            weeks.append((wk, {
                "PassingYards": float(r.get("PassingYards", 0) or 0),
                "PassingTouchdowns": float(r.get("PassingTouchdowns", 0) or 0),
                "PassingInterceptions": float(r.get("PassingInterceptions", 0) or 0),
                "RushingYards": float(r.get("RushingYards", 0) or 0),
                "RushingTouchdowns": float(r.get("RushingTouchdowns", 0) or 0),
                "Receptions": float(r.get("Receptions", 0) or 0),
                "ReceivingYards": float(r.get("ReceivingYards", 0) or 0),
                "ReceivingTouchdowns": float(r.get("ReceivingTouchdowns", 0) or 0),
            }))
        return weeks

    def team_record_string(team_name):
        rec = standings.get(team_name, {"wins":0,"losses":0,"ties":0})
        return f"Record: {rec['wins']}-{rec['losses']}-{rec['ties']}"

    def build_lineup_prompt(team, week):
        # Build per-player payload with 2023 totals, 2024 YTD up to week-1, and last 3 weeks
        ytd_week = max(0, week - 1)
        payload_lines = []
        for p in team["roster"]:
            resolved = p["Name"]  # already full name
            found23, totals23 = summarize_weekly_year(dh, resolved, 2023)
            if not found23:
                totals23 = {k: 0.0 for k in [
                    "PassingYards","PassingTouchdowns","PassingInterceptions","RushingYards","RushingTouchdowns","Receptions","ReceivingYards","ReceivingTouchdowns"
                ]}
            ytd24 = summarize_2024_to_week(resolved, ytd_week) if ytd_week > 0 else {k: 0.0 for k in totals23.keys()}
            last3 = last_3_weeks(resolved, week)
            payload_lines.append(
                f"- {p['Name']} ({p['Position']}): 2023 totals: PY={int(totals23['PassingYards'])}, PTD={int(totals23['PassingTouchdowns'])}, INT={int(totals23['PassingInterceptions'])}, "
                f"RY={int(totals23['RushingYards'])}, RTD={int(totals23['RushingTouchdowns'])}, REC={int(totals23['Receptions'])}, RECY={int(totals23['ReceivingYards'])}, RECTD={int(totals23['ReceivingTouchdowns'])}; "
                f"2024 YTD (weeks 1-{ytd_week if ytd_week>0 else 0}): PY={int(ytd24['PassingYards'])}, PTD={int(ytd24['PassingTouchdowns'])}, INT={int(ytd24['PassingInterceptions'])}, RY={int(ytd24['RushingYards'])}, RTD={int(ytd24['RushingTouchdowns'])}, REC={int(ytd24['Receptions'])}, RECY={int(ytd24['ReceivingYards'])}, RECTD={int(ytd24['ReceivingTouchdowns'])}; "
                f"Last3: " + ", ".join([f"wk{wk}: PY={int(d.get('PassingYards',0))}, PTD={int(d.get('PassingTouchdowns',0))}, INT={int(d.get('PassingInterceptions',0))}, RY={int(d.get('RushingYards',0))}, RTD={int(d.get('RushingTouchdowns',0))}, REC={int(d.get('Receptions',0))}, RECY={int(d.get('ReceivingYards',0))}, RECTD={int(d.get('ReceivingTouchdowns',0))}" for wk,d in last3])
            )

        allowed_names = ", ".join([p["Name"] for p in team["roster"]])
        prompt = (
            "Fantasy Football Start/Sit Decision Setup:\n"
            f"{format_roster_format()}\n"
            f"{format_scoring_format()}\n\n"
            f"Team: {team['name']} | {team_record_string(team['name'])}\n"
            "You are setting a fantasy football lineup. Roster configuration: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE).\n"
            "Respond ONLY with a JSON object of the form {\"qb\": \"Name\", \"rbs\": [\"Name\",\"Name\"], \"wrs\": [\"Name\",\"Name\"], \"te\": \"Name\", \"flex\": \"Name\"}. "
            f"All names MUST be selected from this list: {allowed_names}.\n\n" +
            "Players and stats context:\n" +
            "\n".join(payload_lines)
        )
        return prompt

    def build_trade_context(team):
        # Build stats for every player: 2024 YTD up to previous week and last week
        # Use week_scoped variable later when we call
        return None  # computed inline per call

    # Trade logging setup
    trade_log_dir = out_dir
    os.makedirs(trade_log_dir, exist_ok=True)
    trade_log_path = os.path.join(trade_log_dir, f"llm_calls_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    def log_trade_call(stage, proposer_team, receiver_team, week_num, response):
        record = {
            "time": datetime.now().isoformat(),
            "stage": stage,
            "week": week_num,
            "proposer": proposer_team["name"],
            "receiver": receiver_team["name"],
            "output": response,
        }
        with open(trade_log_path, "a") as lf:
            lf.write(json.dumps(record) + "\n")

    accepted_log_path = os.path.join(trade_log_dir, f"accepted_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    def log_accepted_trade(week_num, proposer_team, receiver_team, give_names, receive_names, accepted_stage):
        rec = {
            "time": datetime.now().isoformat(),
            "week": week_num,
            "accepted_stage": accepted_stage,
            "proposer": proposer_team["name"],
            "receiver": receiver_team["name"],
            "give": list(give_names),
            "receive": list(receive_names),
        }
        with open(accepted_log_path, "a") as af:
            af.write(json.dumps(rec) + "\n")

    def summarize_player_for_trade(full_name, position, upto_week):
        ytd = summarize_2024_to_week(full_name, max(0, upto_week - 1))
        last_week = last_3_weeks(full_name, upto_week)
        last = last_week[-1][1] if last_week else {}
        return (
            f"{full_name} ({position}) | 2024 YTD to wk{max(0, upto_week-1)}: "
            f"PY={int(ytd['PassingYards'])}, PTD={int(ytd['PassingTouchdowns'])}, INT={int(ytd['PassingInterceptions'])}, "
            f"RY={int(ytd['RushingYards'])}, RTD={int(ytd['RushingTouchdowns'])}, REC={int(ytd['Receptions'])}, RECY={int(ytd['ReceivingYards'])}, RECTD={int(ytd['ReceivingTouchdowns'])}; "
            f"Last wk: PY={int(last.get('PassingYards',0))}, PTD={int(last.get('PassingTouchdowns',0))}, INT={int(last.get('PassingInterceptions',0))}, "
            f"RY={int(last.get('RushingYards',0))}, RTD={int(last.get('RushingTouchdowns',0))}, REC={int(last.get('Receptions',0))}, RECY={int(last.get('ReceivingYards',0))}, RECTD={int(last.get('ReceivingTouchdowns',0))}"
        )

    def propose_and_negotiate_trades(teams_list, week):
        # Each team proposes one trade to another team
        for i, proposer in enumerate(teams_list):
            # Build roster summaries
            proposer_lines = [
                summarize_player_for_trade(p["Name"], p["Position"], week)
                for p in proposer["roster"]
            ]
            # Identify weakest slot heuristic (lowest count of eligible starters)
            counts = Counter([p["Position"] for p in proposer["roster"]])
            weakest = min(("QB","RB","WR","TE"), key=lambda pos: counts.get(pos, 0))
            others = [t for t in teams_list if t is not proposer]
            other_blocks = []
            for t in others:
                lines = [summarize_player_for_trade(p["Name"], p["Position"], week) for p in t["roster"]]
                other_blocks.append(f"Team {t['name']} roster:\n" + "\n".join(lines))

            prompt1 = (
                "Fantasy Football Trade Negotiation Setup:\n"
                f"{format_roster_format()}\n"
                f"{format_scoring_format()}\n\n"
                "You are proposing a fantasy football trade.\n"+
                f"Your team: {proposer['name']} ({team_record_string(proposer['name'])}) (weakest position heuristic: {weakest}).\n"+
                "Your roster with stats:\n" + "\n".join(proposer_lines) + "\n\n"+
                "Other teams and their rosters with stats (records shown in headers):\n\n" + "\n\n".join([f"Team {t['name']} ({team_record_string(t['name'])}) roster:\n" + "\n".join([summarize_player_for_trade(p['Name'], p['Position'], week) for p in t['roster']]) for t in others]) + "\n\n"+
                "Propose ONE trade with EXACTLY equal number of players on both sides (choose 1-for-1, 2-for-2, or 3-for-3). Prefer 2-for-2 if both teams have depth; only use 1-for-1 if you cannot form a fair 2-for-2. NEVER propose an uneven trade (e.g., 2-for-1 or 3-for-2). "
                "Include positions and a convincing rationale for the other team. "
                "Return ONLY JSON: {\"target_team\": \"TeamName\", \"give\": [{\"name\": \"Full Name\", \"position\": \"POS\"}, ...], \"receive\": [{\"name\": \"Full Name\", \"position\": \"POS\"}, ...], \"rationale\": \"why this helps them\"}."
            )
            resp1 = llm.call_with_prompt(proposer["model_id"], prompt1)
            log_trade_call("propose", proposer, others[0] if others else proposer, week, resp1)
            try:
                js1 = _extract_json(resp1) if isinstance(resp1, str) else resp1
                if not js1 and isinstance(resp1, str):
                    js1 = _parse_trade_proposal(resp1)
            except Exception:
                js1 = {}
            target_name = (js1 or {}).get("target_team")
            give = (js1 or {}).get("give", [])
            receive = (js1 or {}).get("receive", [])
            # Support give/receive as list of dicts with name/position
            def extract_names(items):
                res = []
                for it in (items or []):
                    if isinstance(it, dict) and it.get("name"):
                        res.append(it.get("name"))
                    elif isinstance(it, str):
                        res.append(it)
                return res
            give_names = extract_names(give)
            receive_names = extract_names(receive)
            target = next((t for t in teams_list if t["name"] == target_name), None)
            # Fallbacks to ensure flow continues
            if not target:
                target = others[0] if others else proposer
            ok, _ = validate_trade_proposal(proposer, target, give_names, receive_names)
            if not ok:
                # Prefer 2-for-2 fallback when possible; else 1-for-1
                if len(proposer["roster"]) >= 2 and len(target["roster"]) >= 2:
                    give_names = [proposer["roster"][0]["Name"], proposer["roster"][1]["Name"]]
                    receive_names = [target["roster"][0]["Name"], target["roster"][1]["Name"]]
                elif proposer["roster"] and target["roster"]:
                    give_names = [proposer["roster"][0]["Name"]]
                    receive_names = [target["roster"][0]["Name"]]

            # Call 2: receiver decides accept/counter
            context_prop = "\n".join(proposer_lines)
            context_recv = "\n".join([summarize_player_for_trade(p["Name"], p["Position"], week) for p in target["roster"]])
            prompt2 = (
                "Fantasy Football Trade Negotiation Setup:\n"
                f"{format_roster_format()}\n"
                f"{format_scoring_format()}\n\n"
                f"You are the receiver of a trade proposal. Your team: {target['name']} ({team_record_string(target['name'])}).\n"+
                f"Proposal: you would RECEIVE {receive_names} and GIVE {give_names}. Rationale from proposer: {(js1 or {}).get('rationale','')}\n"+
                "Proposer roster with stats:\n" + context_prop + "\n\nReceiver roster with stats:\n" + context_recv + "\n\n"+
                "If the proposal clearly benefits your team (improves starters, depth, or expected points), choose accept. If not, respond with a COUNTER of EXACTLY equal size (1-for-1, 2-for-2, or 3-for-3). NEVER propose an uneven trade (e.g., 2-for-1).\n"
                "Respond ONLY JSON. Either accept: {\"decision\": \"accept\"} OR counter: {\"decision\": \"counter\", \"give\": [{\"name\":\"...\",\"position\":\"...\"}], \"receive\": [{\"name\":\"...\",\"position\":\"...\"}], \"rationale\": \"...\"}."
            )
            resp2 = llm.call_with_prompt(target["model_id"], prompt2)
            log_trade_call("counter_or_accept", proposer, target, week, resp2)
            try:
                js2 = _extract_json(resp2) if isinstance(resp2, str) else resp2
                if not js2 and isinstance(resp2, str):
                    js2 = _parse_trade_decision(resp2)
            except Exception:
                js2 = {"decision": "counter", "give": [{"name": receive_names[0] if receive_names else target["roster"][0]["Name"], "position": ""}], "receive": [{"name": give_names[0] if give_names else proposer["roster"][0]["Name"], "position": ""}], "rationale": "Auto-counter due to unparseable response."}
            if (js2 or {}).get("decision") == "accept" and give_names and receive_names and len(give_names) == len(receive_names):
                apply_trade(proposer, target, give_names, receive_names)
                write_team_csv(proposer); write_team_csv(target)
                log_accepted_trade(week, proposer, target, give_names, receive_names, "counter_or_accept")
                continue

            # Call 3: proposer reacts (accept or counter)
            give2 = extract_names((js2 or {}).get("give", []))
            receive2 = extract_names((js2 or {}).get("receive", []))
            ok2, _ = validate_trade_proposal(target, proposer, give2, receive2)
            if not ok2:
                if len(target["roster"]) >= 2 and len(proposer["roster"]) >= 2:
                    give2 = [target["roster"][0]["Name"], target["roster"][1]["Name"]]
                    receive2 = [proposer["roster"][0]["Name"], proposer["roster"][1]["Name"]]
                elif target["roster"] and proposer["roster"]:
                    give2 = [target["roster"][0]["Name"]]
                    receive2 = [proposer["roster"][0]["Name"]]
            prompt3 = (
                "Fantasy Football Trade Negotiation Setup:\n"
                f"{format_roster_format()}\n"
                f"{format_scoring_format()}\n\n"
                f"You are the original proposer evaluating a counter. Your team: {proposer['name']} ({team_record_string(proposer['name'])}).\n"+
                f"Counter proposal: YOU would GIVE {receive2} and RECEIVE {give2}. Rationale from other team: {(js2 or {}).get('rationale','')}\n"+
                "If the counter clearly benefits your team, choose accept. If not, respond with ONE final COUNTER of EXACTLY equal size (1-for-1, 2-for-2, or 3-for-3). NEVER propose an uneven trade.\n"
                "Respond ONLY JSON. Either accept: {\"decision\": \"accept\"} OR counter once more: {\"decision\": \"counter\", \"give\": [{\"name\":\"...\",\"position\":\"...\"}], \"receive\": [{\"name\":\"...\",\"position\":\"...\"}], \"rationale\": \"...\"}."
            )
            resp3 = llm.call_with_prompt(proposer["model_id"], prompt3)
            log_trade_call("proposer_react", proposer, target, week, resp3)
            try:
                js3 = _extract_json(resp3) if isinstance(resp3, str) else resp3
                if not js3 and isinstance(resp3, str):
                    js3 = _parse_trade_decision(resp3)
            except Exception:
                js3 = {"decision": "reject"}
            if (js3 or {}).get("decision") == "accept" and give2 and receive2 and len(give2) == len(receive2):
                apply_trade(target, proposer, give2, receive2)
                write_team_csv(proposer); write_team_csv(target)
                log_accepted_trade(week, target, proposer, give2, receive2, "proposer_react")
                continue

            # Call 4: receiver final decision (must log even if invalid -> forced reject)
            give3 = extract_names((js3 or {}).get("give", []))
            receive3 = extract_names((js3 or {}).get("receive", []))
            ok3, _ = validate_trade_proposal(proposer, target, give3, receive3)
            if not ok3:
                # If invalid, treat as reject in final stage but still log a final decision
                give3 = []
                receive3 = []
            prompt4 = (
                "Fantasy Football Trade Negotiation Setup:\n"
                f"{format_roster_format()}\n"
                f"{format_scoring_format()}\n\n"
                f"Final decision. Evaluate this final counter. Your team: {target['name']} ({team_record_string(target['name'])}).\n"+
                f"Final counter: you would GIVE {give3} and RECEIVE {receive3}.\n"+
                "If this improves your team, choose accept; otherwise reject. NEVER accept or propose uneven trades.\n"
                "Respond ONLY JSON. Either {\"decision\": \"accept\"} or {\"decision\": \"reject\"}."
            )
            resp4 = llm.call_with_prompt(target["model_id"], prompt4)
            log_trade_call("final_decision", proposer, target, week, resp4)
            try:
                js4 = _extract_json(resp4) if isinstance(resp4, str) else resp4
                if not js4 and isinstance(resp4, str):
                    js4 = _parse_trade_decision(resp4)
            except Exception:
                js4 = {"decision": "reject"}
            if (js4 or {}).get("decision") == "accept" and give3 and receive3 and len(give3) == len(receive3):
                apply_trade(proposer, target, give3, receive3)
                write_team_csv(proposer); write_team_csv(target)
                log_accepted_trade(week, proposer, target, give3, receive3, "final_decision")

    # LLM logging setup
    log_dir = out_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"llm_calls_season_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    def log_llm_call(team_name, model_id, week_num, payload, response):
        record = {
            "time": datetime.now().isoformat(),
            "week": week_num,
            "team": team_name,
            "model": model_id,
            "input": payload,
            "output": response,
        }
        with open(log_path, "a") as lf:
            lf.write(json.dumps(record) + "\n")

    def select_lineup_with_llm(team, week, use_llm=True):
        if not use_llm:
            return select_starting_lineup(team["roster"])
        prompt = build_lineup_prompt(team, week)
        resp = llm.call_with_prompt(team["model_id"], prompt)
        log_llm_call(team["name"], team["model_id"], week, {"prompt": prompt}, resp)
        # Try to parse JSON
        parsed = None
        if isinstance(resp, dict):
            parsed = resp
        elif isinstance(resp, str):
            parsed = _extract_json(resp)

        names_by_slot = {"qb": None, "rbs": [], "wrs": [], "te": None, "flex": None}
        if isinstance(parsed, dict):
            names_by_slot["qb"] = parsed.get("qb")
            names_by_slot["rbs"] = parsed.get("rbs", [])
            names_by_slot["wrs"] = parsed.get("wrs", [])
            names_by_slot["te"] = parsed.get("te")
            names_by_slot["flex"] = parsed.get("flex")

        # Map names to roster entries, filter invalid
        name_to_player = {p["Name"]: p for p in team["roster"]}
        starters = []
        used = set()
        # QB
        if names_by_slot["qb"] in name_to_player and name_to_player[names_by_slot["qb"]]["Position"] == "QB":
            starters.append(name_to_player[names_by_slot["qb"]]); used.add(names_by_slot["qb"])
        # RBs
        for nm in names_by_slot["rbs"]:
            if nm in name_to_player and name_to_player[nm]["Position"] == "RB" and nm not in used and len([p for p in starters if p["Position"] == "RB"]) < 2:
                starters.append(name_to_player[nm]); used.add(nm)
        # WRs
        for nm in names_by_slot["wrs"]:
            if nm in name_to_player and name_to_player[nm]["Position"] == "WR" and nm not in used and len([p for p in starters if p["Position"] == "WR"]) < 2:
                starters.append(name_to_player[nm]); used.add(nm)
        # TE
        if names_by_slot["te"] in name_to_player and name_to_player[names_by_slot["te"]]["Position"] == "TE" and names_by_slot["te"] not in used:
            starters.append(name_to_player[names_by_slot["te"]]); used.add(names_by_slot["te"])
        # FLEX
        flex_nm = names_by_slot["flex"]
        if flex_nm in name_to_player and name_to_player[flex_nm]["Position"] in ("RB","WR","TE") and flex_nm not in used:
            starters.append(name_to_player[flex_nm]); used.add(flex_nm)

        # Fill any missing slots with heuristic
        need_counts = {"QB":1, "RB":2, "WR":2, "TE":1, "FLEX":1}
        counts = Counter([p["Position"] for p in starters])
        # Core positions
        for pos, need in [("QB",1),("RB",2),("WR",2),("TE",1)]:
            while counts.get(pos,0) < need:
                for p in team["roster"]:
                    if p["Position"] == pos and p["Name"] not in used:
                        starters.append(p); used.add(p["Name"])
                        counts[pos] = counts.get(pos,0)+1
                        break
                else:
                    break
        # FLEX
        if len(starters) < 7:
            for p in team["roster"]:
                if p["Name"] in used:
                    continue
                if p["Position"] in ("RB","WR","TE"):
                    starters.append(p); used.add(p["Name"]) 
                    break

        bench = [p for p in team["roster"] if p["Name"] not in used]
        return starters, bench

    for week in range(1, 18):
        # Trades only every 3 weeks: 1,4,7,...
        if (week - 1) % 3 == 0:
            propose_and_negotiate_trades(teams, week)
        # Random matchups
        idxs = list(range(len(teams)))
        random.shuffle(idxs)
        matchups = [(idxs[i], idxs[i+1]) for i in range(0, len(idxs), 2)]

        # Compute results
        week_rows = []
        for a_idx, b_idx in matchups:
            ta = teams[a_idx]
            tb = teams[b_idx]
            # Get LLM-decided lineups
            starters_a, bench_a = select_lineup_with_llm(ta, week, use_llm=True)
            starters_b, bench_b = select_lineup_with_llm(tb, week, use_llm=True)
            ta_points = 0.0
            tb_points = 0.0

            # Per-player points
            for p in starters_a:
                full = p["Name"]  # already full name
                pts = player_week_points(dh, full, week)
                ta_points += pts
                week_rows.append({
                    "Week": week,
                    "Team": ta["name"],
                    "Model": ta["model_id"],
                    "Opponent": tb["name"],
                    "Player": p["Name"],
                    "ResolvedName": full,
                    "Position": p["Position"],
                    "Points": f"{pts:.2f}",
                })
            for p in starters_b:
                full = p["Name"]  # already full name
                pts = player_week_points(dh, full, week)
                tb_points += pts
                week_rows.append({
                    "Week": week,
                    "Team": tb["name"],
                    "Model": tb["model_id"],
                    "Opponent": ta["name"],
                    "Player": p["Name"],
                    "ResolvedName": full,
                    "Position": p["Position"],
                    "Points": f"{pts:.2f}",
                })

            # Final scores
            if ta_points > tb_points:
                standings[ta["name"]]["wins"] += 1
                standings[tb["name"]]["losses"] += 1
            elif tb_points > ta_points:
                standings[tb["name"]]["wins"] += 1
                standings[ta["name"]]["losses"] += 1
            else:
                standings[ta["name"]]["ties"] += 1
                standings[tb["name"]]["ties"] += 1

            standings[ta["name"]]["points_for"] += ta_points
            standings[tb["name"]]["points_for"] += tb_points

            # Append matchup summary rows
            week_rows.append({
                "Week": week,
                "Team": ta["name"],
                "Model": ta["model_id"],
                "Opponent": tb["name"],
                "Player": "TOTAL",
                "ResolvedName": "",
                "Position": "",
                "Points": f"{ta_points:.2f}",
            })
            week_rows.append({
                "Week": week,
                "Team": tb["name"],
                "Model": tb["model_id"],
                "Opponent": ta["name"],
                "Player": "TOTAL",
                "ResolvedName": "",
                "Position": "",
                "Points": f"{tb_points:.2f}",
            })

        # Dump weekly results
        out_week = os.path.join(out_dir, f"week_{week}_results.csv")
        with open(out_week, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Week", "Team", "Model", "Opponent", "Player", "ResolvedName", "Position", "Points"])
            writer.writeheader()
            for r in week_rows:
                writer.writerow(r)

    # Final standings
    standings_rows = []
    for team_name, rec in standings.items():
        standings_rows.append({
            "Team": team_name,
            "Wins": rec["wins"],
            "Losses": rec["losses"],
            "Ties": rec["ties"],
            "PointsFor": f"{rec['points_for']:.2f}",
        })
    # Sort by wins, then points_for
    standings_rows.sort(key=lambda x: (x["Wins"], float(x["PointsFor"])), reverse=True)

    out_standings = os.path.join(out_dir, "final_standings_2024.csv")
    with open(out_standings, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Team", "Wins", "Losses", "Ties", "PointsFor"])
        writer.writeheader()
        for r in standings_rows:
            writer.writerow(r)

    print(f"Saved weekly results to {out_dir} and final standings to {out_standings}")


if __name__ == "__main__":
    simulate_season()


