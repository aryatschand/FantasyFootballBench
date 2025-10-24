#!/usr/bin/env python3
import os
import csv
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DRAFT_DIR = os.path.join(BASE, "draft_results")
SEASON_DIR = os.path.join(BASE, "season_results")
FIG_DIR = os.path.join(BASE, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------
# Helpers for trade logs
# ---------------------
def _normalize_output_to_dict(output):
    if isinstance(output, dict):
        return output
    if not isinstance(output, str):
        return {}
    text = output.strip()
    # strip code fences
    if text.startswith("```"):
        text = text.strip('`')
        # remove optional leading language
        if text.startswith("json\n"):
            text = text[len("json\n"):]
    text = text.strip()
    # attempt JSON parse on the first { .. }
    try:
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
        # try to find a JSON object substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    except Exception:
        pass
    # naive decision extraction
    lowered = text.lower()
    decision = None
    if "\"decision\"" in lowered:
        if "accept" in lowered:
            decision = "accept"
        elif "reject" in lowered:
            decision = "reject"
        elif "counter" in lowered:
            decision = "counter"
    return {"decision": decision} if decision else {}

def _trade_actor(call: dict) -> str:
    stage = str(call.get("stage", ""))
    proposer = call.get("proposer")
    receiver = call.get("receiver")
    if "counter_or_accept" in stage:
        return receiver or proposer or call.get("team", "")
    if "proposer_react" in stage or "propose" in stage:
        return proposer or receiver or call.get("team", "")
    if "final_decision" in stage:
        # Often authored by the proposer in our logs; fall back appropriately
        return call.get("proposer") or call.get("receiver") or call.get("team", "")
    return proposer or receiver or call.get("team", "")

def _trade_decision_label(call: dict) -> str:
    stage = str(call.get("stage", ""))
    output = _normalize_output_to_dict(call.get("output"))
    decision = str(output.get("decision", "")).lower()
    if "counter_or_accept" in stage:
        if decision == "accept":
            return "accept"
        if decision == "counter":
            return "counter"
    if "proposer_react" in stage:
        return "counter"
    if "final_decision" in stage:
        if decision == "accept":
            return "accept"
        if decision == "reject":
            return "reject"
    return ""

def read_final_standings():
    path = os.path.join(SEASON_DIR, "final_standings_2024.csv")
    standings = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if not row.get("Team"):
                continue
            standings.append({
                "team": row["Team"],
                "wins": int(row["Wins"]),
                "losses": int(row["Losses"]),
                "ties": int(row.get("Ties", 0) or 0),
                "points_for": float(row["PointsFor"]),
            })
    return standings

def read_weekly_results():
    week_files = [f for f in os.listdir(SEASON_DIR) if f.startswith("week_") and f.endswith("_results.csv")]
    data = []
    for wf in sorted(week_files, key=lambda x: int(x.split("_")[1])):
        path = os.path.join(SEASON_DIR, wf)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("Player") or "").strip().upper() == "TOTAL":
                    data.append({
                        "week": int(row["Week"]),
                        "team": row["Team"],
                        "model": row.get("Model"),
                        "opponent": row["Opponent"],
                        "points": float(row.get("Points", 0) or 0),
                    })
    return data

def read_accepted_trades():
    # pick the accepted_trades*.jsonl if exists
    files = [f for f in os.listdir(SEASON_DIR) if f.startswith("accepted_trades_") and f.endswith(".jsonl")]
    trades = []
    for fn in files:
        path = os.path.join(SEASON_DIR, fn)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trades.append(json.loads(line))
                except Exception:
                    # Some files might be csv-like, skip lines not json
                    continue
    return trades

def read_trade_calls():
    files = [f for f in os.listdir(SEASON_DIR) if f.startswith("llm_calls_trades_") and f.endswith(".jsonl")]
    calls = []
    for fn in files:
        path = os.path.join(SEASON_DIR, fn)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    calls.append(json.loads(line))
                except Exception:
                    continue
    return calls

def read_draft_team_csvs():
    # team draft CSVs: one per team
    players_by_team = {}
    for fn in os.listdir(DRAFT_DIR):
        if fn.endswith("_2024.csv") and fn != "TopPlayers_2024_Projections_PPR.csv":
            path = os.path.join(DRAFT_DIR, fn)
            team = fn.rsplit("_2024.csv", 1)[0]
            roster = []
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Name")
                    pos = row.get("Position")
                    proj = row.get("FantasyPoints")
                    try:
                        proj = float(proj) if proj not in (None, "") else None
                    except Exception:
                        proj = None
                    if name:
                        roster.append({"name": name, "position": pos, "proj": proj})
            players_by_team[team] = roster
    return players_by_team

def read_top_projections():
    # From draft_results/TopPlayers_2024_Projections_PPR.csv
    proj_map = {}
    top_path = os.path.join(DRAFT_DIR, "TopPlayers_2024_Projections_PPR.csv")
    if not os.path.exists(top_path):
        return proj_map
    with open(top_path, newline="") as f:
        reader = csv.DictReader(f)
        # Try to find the projection column by common names
        for row in reader:
            nm = normalize_name(row.get("Name"))
            if not nm:
                continue
            val = None
            for key in ("FantasyPointsPPR", "FantasyPoints", "ProjPointsPPR", "ProjPoints"):
                if key in row and row.get(key) not in (None, ""):
                    try:
                        val = float(row.get(key))
                        break
                    except Exception:
                        continue
            if val is not None:
                proj_map[nm] = val
    return proj_map

def cumulative_points_by_team(weekly):
    weeks = sorted({r["week"] for r in weekly})
    teams = sorted({r["team"] for r in weekly})
    cum = {t: [] for t in teams}
    totals = {t: 0.0 for t in teams}
    for w in weeks:
        for t in teams:
            pts = sum(r["points"] for r in weekly if r["week"] == w and r["team"] == t)
            totals[t] += pts
            cum[t].append(totals[t])
    return weeks, cum

def win_loss_trend(weekly):
    # Win/loss per week
    results_by_week = defaultdict(list)
    for r in weekly:
        results_by_week[r["week"]].append(r)
    trend = defaultdict(list)  # team -> list cumulative wins
    cum_wins = Counter()
    for w in sorted(results_by_week):
        rows = results_by_week[w]
        # pair rows by opponents
        by_team = {(x["team"], x["opponent"]): x for x in rows}
        seen = set()
        for x in rows:
            team = x["team"]
            opp = x["opponent"]
            key = tuple(sorted([team, opp]))
            if key in seen:
                continue
            seen.add(key)
            a = by_team.get((team, opp))
            b = by_team.get((opp, team))
            if not a or not b:
                continue
            if a["points"] > b["points"]:
                cum_wins[a["team"]] += 1
            elif b["points"] > a["points"]:
                cum_wins[b["team"]] += 1
        # record snapshot this week
        teams = {x["team"] for x in rows}
        for t in teams:
            trend[t].append(cum_wins[t])
    return sorted(results_by_week), trend

def bar_total_points_vs_wins(standings):
    teams = [s["team"] for s in standings]
    wins = [s["wins"] for s in standings]
    pts = [s["points_for"] for s in standings]
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()
    ax1.bar(teams, wins, color="#4e79a7", alpha=0.7, label="Wins")
    ax2.plot(teams, pts, color="#f28e2c", marker="o", label="Points For")
    ax1.set_title("Wins vs Points For by Team")
    ax1.set_ylabel("Wins")
    ax2.set_ylabel("Points For")
    ax1.set_xticklabels(teams, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_wins_vs_points.png"))
    plt.close(fig)

def line_cumulative_points(weekly):
    weeks, cum = cumulative_points_by_team(weekly)
    fig, ax = plt.subplots(figsize=(10,6))
    for team, vals in cum.items():
        ax.plot(weeks, vals, label=team)
    ax.set_title("Cumulative Points by Team over Season")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative Points")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_cumulative_points.png"))
    plt.close(fig)

def line_cumulative_wins(weekly):
    weeks, trend = win_loss_trend(weekly)
    fig, ax = plt.subplots(figsize=(10,6))
    for team, vals in trend.items():
        ax.plot(list(range(1, len(vals)+1)), vals, label=team)
    ax.set_title("Cumulative Wins by Team over Season")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative Wins")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_cumulative_wins.png"))
    plt.close(fig)

def box_weekly_points_distribution(weekly):
    by_team = defaultdict(list)
    for r in weekly:
        by_team[r["team"]].append(r["points"])
    data = [by_team[t] for t in sorted(by_team)]
    labels = sorted(by_team)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.boxplot(data, labels=labels, vert=True)
    ax.set_title("Weekly Points Distribution by Team")
    ax.set_ylabel("Points")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_weekly_points_boxplot.png"))
    plt.close(fig)

def draft_position_breakdown(players_by_team):
    # Stacked bar of roster positions drafted per team
    positions = ["QB","RB","WR","TE"]
    teams = sorted(players_by_team)
    counts = {pos: [0]*len(teams) for pos in positions}
    for i, team in enumerate(teams):
        pos_counts = Counter(p["position"] for p in players_by_team[team])
        for pos in positions:
            counts[pos][i] = pos_counts.get(pos, 0)
    fig, ax = plt.subplots(figsize=(10,6))
    bottom = [0]*len(teams)
    colors = {"QB":"#1f77b4","RB":"#2ca02c","WR":"#ff7f0e","TE":"#9467bd"}
    for pos in positions:
        ax.bar(teams, counts[pos], bottom=bottom, label=pos, color=colors[pos])
        bottom = [b+c for b,c in zip(bottom, counts[pos])]
    ax.set_title("Drafted Roster Composition by Position")
    ax.set_ylabel("Count of Drafted Players")
    ax.set_xticklabels(teams, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_draft_position_breakdown.png"))
    plt.close(fig)

def draft_projected_points(players_by_team):
    teams = sorted(players_by_team)
    sums = []
    for team in teams:
        proj_sum = sum(p["proj"] for p in players_by_team[team] if isinstance(p["proj"], (int,float)))
        sums.append(proj_sum)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(teams, sums, color="#59a14f")
    ax.set_title("Sum of Draft Projected Points by Team")
    ax.set_ylabel("Projected Points (sum)")
    ax.set_xticklabels(teams, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "06_draft_projected_points.png"))
    plt.close(fig)

def trade_volume_by_week(trade_calls):
    by_week = Counter(c.get("week") for c in trade_calls if isinstance(c, dict) and c.get("week") is not None)
    weeks = sorted(by_week)
    counts = [by_week[w] for w in weeks]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(weeks, counts, marker="o")
    ax.set_title("Trade Call Volume by Week")
    ax.set_xlabel("Week")
    ax.set_ylabel("# Trade Calls")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "07_trade_calls_by_week.png"))
    plt.close(fig)

def accepted_trades_by_team(trades):
    by_team = Counter()
    for t in trades:
        proposer = t.get("proposer")
        receiver = t.get("receiver")
        if proposer:
            by_team[proposer] += 1
        if receiver:
            by_team[receiver] += 1
    teams = sorted(by_team)
    counts = [by_team[t] for t in teams]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(teams, counts, color="#e15759")
    ax.set_title("Accepted Trades Involving Each Team")
    ax.set_ylabel("# Accepted Trades")
    ax.set_xticklabels(teams, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "08_accepted_trades_by_team.png"))
    plt.close(fig)

def model_performance_scatter(weekly):
    # scatter of avg weekly points vs total wins per team
    totals = defaultdict(float)
    games = Counter()
    for r in weekly:
        totals[r["team"]] += r["points"]
        games[r["team"]] += 1
    avg_pts = {t: (totals[t]/games[t] if games[t] else 0.0) for t in totals}
    # recompute wins with helper
    weeks, trend = win_loss_trend(weekly)
    final_wins = {t: (vals[-1] if vals else 0) for t, vals in trend.items()}
    xs = []
    ys = []
    labels = []
    for t in sorted(avg_pts):
        xs.append(avg_pts[t])
        ys.append(final_wins.get(t, 0))
        labels.append(t)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(xs, ys)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (xs[i], ys[i]), fontsize=8, alpha=0.7)
    ax.set_title("Average Weekly Points vs Final Wins")
    ax.set_xlabel("Avg Weekly Points")
    ax.set_ylabel("Final Wins")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "09_avg_points_vs_wins.png"))
    plt.close(fig)

def head_to_head_heatmap(weekly):
    teams = sorted({r["team"] for r in weekly})
    idx = {t:i for i,t in enumerate(teams)}
    # wins matrix
    n = len(teams)
    mat = [[0]*n for _ in range(n)]
    # derive wins by head-to-head
    by_week = defaultdict(list)
    for r in weekly:
        by_week[r["week"]].append(r)
    for w in sorted(by_week):
        rows = by_week[w]
        pair = {(x["team"], x["opponent"]): x for x in rows}
        seen = set()
        for x in rows:
            a = x["team"]
            b = x["opponent"]
            key = tuple(sorted([a,b]))
            if key in seen:
                continue
            seen.add(key)
            ra = pair.get((a,b)); rb = pair.get((b,a))
            if not ra or not rb:
                continue
            if ra["points"] > rb["points"]:
                mat[idx[a]][idx[b]] += 1
            elif rb["points"] > ra["points"]:
                mat[idx[b]][idx[a]] += 1
    # simple heatmap via imshow
    fig, ax = plt.subplots(figsize=(8,7))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(teams, rotation=45, ha="right")
    ax.set_yticklabels(teams)
    ax.set_title("Head-to-Head Wins Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "10_head_to_head_heatmap.png"))
    plt.close(fig)

def trade_negotiation_patterns(trade_calls):
    # Analyze how LLMs respond to trade proposals (accept, counter, reject)
    responses = defaultdict(lambda: {"accept": 0, "counter": 0, "reject": 0})
    for call in trade_calls:
        if not isinstance(call, dict):
            continue
        if not call.get("stage"):
            continue
        actor = _trade_actor(call)
        actor = (actor or call.get("team", "")).replace("_Team_", "").replace("_", " ")
        decision = _trade_decision_label(call)
        if decision in ("accept", "counter", "reject"):
            responses[actor][decision] += 1

    teams = list(responses.keys()) or ["No Data"]
    accept = [responses[t]["accept"] for t in teams]
    counter = [responses[t]["counter"] for t in teams]
    reject = [responses[t]["reject"] for t in teams]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(teams))
    ax.bar(x, accept, label="Accept", color="#4CAF50")
    ax.bar(x, counter, bottom=accept, label="Counter", color="#FF9800")
    ax.bar(x, reject, bottom=[a+c for a,c in zip(accept, counter)], label="Reject", color="#F44336")

    ax.set_title("Trade Negotiation Response Patterns by Model")
    ax.set_ylabel("Number of Responses")
    ax.set_xticks(range(len(teams)))
    ax.set_xticklabels(teams, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "11_trade_negotiation_patterns.png"))
    plt.close(fig)

def draft_strategy_value_vs_projections(draft):
    # Analyze if LLMs draft players with high projections vs. team-building strategy
    # Compare drafted players' projected points vs. actual performance rank
    # This would require more complex analysis - for now, show distribution of projected points by position
    positions = ["QB", "RB", "WR", "TE"]
    pos_data = {pos: [] for pos in positions}

    for team, players in draft.items():
        for player in players:
            if player["position"] in positions and player["proj"]:
                pos_data[player["position"]].append(player["proj"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, pos in enumerate(positions):
        if pos_data[pos]:
            axes[i].hist(pos_data[pos], bins=10, alpha=0.7, color="#8B4513")
            axes[i].set_title(f"{pos} Drafted Player Projected Points")
            axes[i].set_xlabel("Projected Points")
            axes[i].set_ylabel("Frequency")
        else:
            axes[i].text(0.5, 0.5, f"No {pos} data", ha="center", va="center")
            axes[i].set_title(f"{pos} Drafted Player Projected Points")

    fig.suptitle("Draft Strategy: Projected Points Distribution by Position")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "12_draft_strategy_value_vs_projections.png"))
    plt.close(fig)

def start_sit_recent_vs_historical():
    # Analyze how LLMs weight recent performance vs historical stats in start/sit decisions
    # Parse season logs to see start/sit patterns
    season_logs = []
    for fn in os.listdir(SEASON_DIR):
        if fn.startswith("llm_calls_season_"):
            with open(os.path.join(SEASON_DIR, fn)) as f:
                for line in f:
                    try:
                        season_logs.append(json.loads(line))
                    except:
                        continue

    recent_focus = {"high_recent": 0, "balanced": 0, "historical_focus": 0}

    for log in season_logs:
        if "rationale" in str(log.get("output", "")):
            rationale = str(log.get("output", ""))
            if "week 1" in rationale or "recent" in rationale.lower():
                recent_focus["high_recent"] += 1
            elif "2023" in rationale or "historical" in rationale.lower():
                recent_focus["historical_focus"] += 1
            else:
                recent_focus["balanced"] += 1

    labels = list(recent_focus.keys())
    values = list(recent_focus.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    ax.set_title("Start/Sit Decision Patterns: Recent vs Historical Performance Weighting")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "13_start_sit_recent_vs_historical.png"))
    plt.close(fig)

def trade_acceptance_rates_by_model(trade_calls):
    # Calculate acceptance rates for each model in trade negotiations
    model_stats = defaultdict(lambda: {"total_proposals": 0, "accepted": 0, "rejected": 0})

    for call in trade_calls:
        if not isinstance(call, dict):
            continue
        stage = str(call.get("stage", ""))
        actor = _trade_actor(call)
        actor = (actor or call.get("team", "")).replace("_Team_", "").replace("_", " ")
        if "propose" in stage:
            model_stats[actor]["total_proposals"] += 1
        elif "final_decision" in stage:
            decision = _normalize_output_to_dict(call.get("output")).get("decision", "")
            decision = str(decision).lower()
            if decision == "accept":
                model_stats[actor]["accepted"] += 1
            elif decision == "reject":
                model_stats[actor]["rejected"] += 1

    teams = [t for t in model_stats.keys() if model_stats[t]["total_proposals"] > 0] or ["No Data"]
    acceptance_rates = [(model_stats[t]["accepted"] / model_stats[t]["total_proposals"]) * 100 if model_stats[t]["total_proposals"] > 0 else 0 for t in teams]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(teams, acceptance_rates, color="#9C27B0")
    ax.set_title("Trade Acceptance Rates by Model")
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_xticklabels(teams, rotation=45, ha="right")

    for bar, rate in zip(bars, acceptance_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "14_trade_acceptance_rates_by_model.png"))
    plt.close(fig)

def player_selection_patterns_by_model(draft):
    # Show how different models prioritize different positions in drafting
    model_positions = defaultdict(lambda: defaultdict(int))

    for team, players in draft.items():
        model = team.replace("_Team_", "").replace("_", " ")
        for player in players:
            model_positions[model][player["position"]] += 1

    models = list(model_positions.keys())
    positions = ["QB", "RB", "WR", "TE"]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(models))
    width = 0.2

    for i, pos in enumerate(positions):
        counts = [model_positions[m].get(pos, 0) for m in models]
        ax.bar([xi + i*width for xi in x], counts, width, label=pos)

    ax.set_title("Player Selection Patterns: Position Preferences by Model")
    ax.set_ylabel("Number of Players Drafted")
    ax.set_xticks([xi + 1.5*width for xi in x])
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "15_player_selection_patterns_by_model.png"))
    plt.close(fig)

def negotiation_complexity_analysis(trade_calls):
    # Analyze complexity of trade negotiations (number of rounds per trade)
    trade_complexity = defaultdict(int)

    # Group by trade week and teams involved
    active_trades = defaultdict(list)

    for call in trade_calls:
        if isinstance(call, dict):
            week = call.get("week")
            proposer = call.get("proposer")
            receiver = call.get("receiver")
            if proposer and receiver:
                trade_key = f"{week}_{min(proposer, receiver)}_{max(proposer, receiver)}"
                active_trades[trade_key].append(call)

    for trade_key, calls in active_trades.items():
        # Count stages: propose -> counter -> final_decision
        stages = set()
        for call in calls:
            stage = call.get("stage", "")
            if "propose" in stage:
                stages.add("propose")
            elif "counter" in stage:
                stages.add("counter")
            elif "final_decision" in stage:
                stages.add("final")
        complexity = len(stages)
        trade_complexity[complexity] += 1

    complexities = sorted(trade_complexity.keys())
    counts = [trade_complexity[c] for c in complexities]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(complexities, counts, color="#FF5722")
    ax.set_title("Trade Negotiation Complexity: Rounds per Trade")
    ax.set_xlabel("Number of Negotiation Rounds")
    ax.set_ylabel("Number of Trades")
    ax.set_xticks(complexities)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "16_negotiation_complexity_analysis.png"))
    plt.close(fig)

def strategic_trade_value_analysis(trades):
    # Analyze what types of players are traded (starters vs bench, positions)
    traded_positions = defaultdict(int)

    for trade in trades:
        if not isinstance(trade, dict):
            continue
        # accept logs are accepted_trades .jsonl with plain fields
        give = trade.get("give") or []
        receive = trade.get("receive") or []
        # Sometimes lists of strings
        def add_player(p):
            if isinstance(p, dict):
                pos = p.get("position", "")
            else:
                # try to infer position from name string (unknown -> Other)
                pos = "Other"
            traded_positions[pos] += 1
        for p in give:
            add_player(p)
        for p in receive:
            add_player(p)

    positions = sorted(traded_positions.keys()) or ["No Data"]
    counts = [traded_positions[pos] for pos in positions]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(positions, counts, color="#3F51B5")
    ax.set_title("Strategic Trade Analysis: Positions Most Frequently Traded")
    ax.set_ylabel("Number of Players Traded")
    ax.set_xlabel("Position")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "17_strategic_trade_value_analysis.png"))
    plt.close(fig)

def model_specific_behavior_patterns(weekly):
    # Analyze consistency in performance and decision patterns by model
    # Group weekly results by model
    model_performance = defaultdict(list)

    for result in weekly:
        team = result["team"]
        model = team.split("_Team_")[0] if "_Team_" in team else team
        model_performance[model].append(result["points"])

    models = list(model_performance.keys())
    avg_points = [sum(pts)/len(pts) for pts in model_performance.values()]
    std_devs = [sum((x - avg)**2 for x in pts)**0.5 / len(pts) if len(pts) > 1 else 0
                for pts, avg in zip(model_performance.values(), avg_points)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(models))
    ax.bar(x, avg_points, yerr=std_devs, capsize=5, color="#607D8B", alpha=0.7)
    ax.set_title("Model-Specific Performance Patterns: Average Points with Variability")
    ax.set_ylabel("Average Weekly Points")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "18_model_specific_behavior_patterns.png"))
    plt.close(fig)

def decision_consistency_over_time(trade_calls):
    # Track how consistent models are in their trade decisions over time
    model_decisions = defaultdict(lambda: defaultdict(int))

    for call in trade_calls:
        if not isinstance(call, dict) or not call.get("stage"):
            continue
        team = _trade_actor(call)
        team = (team or call.get("team", "")).replace("_Team_", "").replace("_", " ")
        week = call.get("week", 0)
        label = _trade_decision_label(call)
        if label == "accept":
            model_decisions[team][week] = 1
        elif label == "reject":
            model_decisions[team][week] = -1
        elif label == "counter":
            model_decisions[team][week] = 0

    # Calculate consistency score (variance in decision types)
    consistency_scores = {}
    for model, decisions in model_decisions.items():
        if len(decisions) > 1:
            values = list(decisions.values())
            mean = sum(values) / len(values)
            variance = sum((x - mean)**2 for x in values) / len(values)
            consistency_scores[model] = variance  # Lower variance = more consistent

    models = sorted(consistency_scores.keys()) or ["No Data"]
    scores = [consistency_scores[m] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(models, scores, color="#795548")
    ax.set_title("Decision Consistency Over Time: Trade Behavior Variance")
    ax.set_ylabel("Decision Variance (Lower = More Consistent)")
    ax.set_xticklabels(models, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "19_decision_consistency_over_time.png"))
    plt.close(fig)

def performance_correlation_with_trading_activity(weekly, trade_calls):
    # Correlate team performance with their trading frequency
    trade_activity = defaultdict(int)
    for call in trade_calls:
        if not isinstance(call, dict):
            continue
        team = _trade_actor(call)
        team = (team or call.get("team", "")).replace("_Team_", "").replace("_", " ")
        trade_activity[team] += 1

    performance_data = defaultdict(list)
    for result in weekly:
        team = result["team"]
        performance_data[team].append(result["points"])

    avg_performance = {team: sum(pts)/len(pts) for team, pts in performance_data.items()}

    # Scatter plot of trading activity vs average performance
    keys = list(set(trade_activity.keys()) & set(avg_performance.keys()))
    if not keys:
        # graceful empty plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No overlapping data to correlate", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "20_performance_correlation_with_trading_activity.png"))
        plt.close(fig)
        return

    teams = keys
    trades = [trade_activity.get(t, 0) for t in teams]
    performance = [avg_performance.get(t, 0.0) for t in teams]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(trades, performance, s=100, alpha=0.7, color="#009688")

    for i, team in enumerate(teams):
        ax.annotate(team, (trades[i], performance[i]), fontsize=8, alpha=0.8)

    ax.set_title("Performance Correlation with Trading Activity")
    ax.set_xlabel("Number of Trade Calls")
    ax.set_ylabel("Average Weekly Points")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "20_performance_correlation_with_trading_activity.png"))
    plt.close(fig)

#############################
# Correctness-focused graphs #
#############################

REPO_ROOT = os.path.abspath(os.path.join(BASE, os.pardir, os.pardir, os.pardir))
DATA_2024_DIR = os.path.join(REPO_ROOT, "data", "2024")

def normalize_name(name: str) -> str:
    return (name or "").strip().lower()

def read_player_week_points():
    # Build mapping: name_lower -> {week -> PPR points} and name_lower -> fantasy position (last seen)
    name_to_week_points = defaultdict(dict)
    name_to_position = {}
    for wk in range(1, 18):
        week_dir = os.path.join(DATA_2024_DIR, str(wk))
        path = os.path.join(week_dir, "PlayerGameStatsByWeek.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nm = normalize_name(row.get("Name"))
                if not nm:
                    continue
                try:
                    ppr = float(row.get("FantasyPointsPPR", 0) or 0)
                except Exception:
                    ppr = 0.0
                name_to_week_points[nm][wk] = ppr
                pos = row.get("FantasyPosition") or row.get("Position")
                if pos:
                    name_to_position[nm] = pos
    return name_to_week_points, name_to_position

def read_player_season_points():
    # Map: name_lower -> season PPR
    path = os.path.join(DATA_2024_DIR, "PlayerSeasonStats.csv")
    season_points = {}
    if not os.path.exists(path):
        return season_points
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nm = normalize_name(row.get("Name"))
            if not nm:
                continue
            try:
                ppr = float(row.get("FantasyPointsPPR", 0) or 0)
            except Exception:
                ppr = 0.0
            season_points[nm] = ppr
    return season_points

def compute_trade_values(trades, name_to_week_points):
    # Returns: per_team_value, per_trade_details
    per_team = defaultdict(float)
    details = []  # list of dict with trade index, week, proposer_value, receiver_value
    for idx, t in enumerate(trades):
        if not isinstance(t, dict):
            continue
        week = int(t.get("week", 0) or 0)
        proposer = t.get("proposer")
        receiver = t.get("receiver")
        give = t.get("give") or []
        receive = t.get("receive") or []

        def sum_points(players, start_week):
            total = 0.0
            for p in players:
                nm = p.get("name") if isinstance(p, dict) else p
                nm = normalize_name(nm)
                week_map = name_to_week_points.get(nm, {})
                for wk in range(start_week, 18):
                    total += float(week_map.get(wk, 0.0))
            return total

        proposer_gain = sum_points(receive, week) - sum_points(give, week)
        receiver_gain = sum_points(give, week) - sum_points(receive, week)

        if proposer:
            per_team[proposer] += proposer_gain
        if receiver:
            per_team[receiver] += receiver_gain

        details.append({
            "idx": idx+1,
            "week": week,
            "proposer": proposer,
            "receiver": receiver,
            "proposer_gain": proposer_gain,
            "receiver_gain": receiver_gain,
        })
    return per_team, details

def trade_value_by_team_graph(trades, name_to_week_points):
    per_team, _ = compute_trade_values(trades, name_to_week_points)
    teams = list(per_team.keys()) or ["No Trades"]
    values = [per_team.get(t, 0.0) for t in teams]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(teams, values, color="#0072B2")
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_title("Cumulative Trade Value by Team (Rest-of-Season PPR)")
    ax.set_ylabel("Points Gained From Trades")
    ax.set_xticklabels(teams, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "21_trade_value_by_team.png"))
    plt.close(fig)

def trade_value_per_trade_graph(trades, name_to_week_points):
    _, details = compute_trade_values(trades, name_to_week_points)
    if not details:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No accepted trades", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(os.path.join(FIG_DIR, "22_trade_value_per_trade.png"))
        plt.close(fig)
        return
    labels = [f"T{d['idx']}@W{d['week']}" for d in details]
    prop_vals = [d["proposer_gain"] for d in details]
    recv_vals = [d["receiver_gain"] for d in details]
    x = range(len(details))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width/2 for i in x], prop_vals, width, label="Proposer", color="#59a14f")
    ax.bar([i + width/2 for i in x], recv_vals, width, label="Receiver", color="#e15759")
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_title("Trade Value per Trade (Rest-of-Season PPR)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Points Gained")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "22_trade_value_per_trade.png"))
    plt.close(fig)

def wins_after_trade_graph(weekly, trades):
    # Compute wins after trade week for both teams
    # Build per-week win map using earlier logic
    results_by_week = defaultdict(list)
    for r in weekly:
        results_by_week[r["week"]].append(r)
    team_wins_cum = defaultdict(int)
    team_wins_by_week = defaultdict(dict)  # team -> week -> cum wins up to week
    for w in sorted(results_by_week):
        rows = results_by_week[w]
        pair = {(x["team"], x["opponent"]): x for x in rows}
        seen = set()
        for x in rows:
            a = x["team"]; b = x["opponent"]
            key = tuple(sorted([a,b]))
            if key in seen:
                continue
            seen.add(key)
            ra = pair.get((a,b)); rb = pair.get((b,a))
            if not ra or not rb:
                continue
            if ra["points"] > rb["points"]:
                team_wins_cum[a] += 1
            elif rb["points"] > ra["points"]:
                team_wins_cum[b] += 1
        for x in rows:
            team_wins_by_week[x["team"]][w] = team_wins_cum[x["team"]]

    labels = []
    prop_wins = []
    recv_wins = []
    for idx, t in enumerate(trades):
        if not isinstance(t, dict):
            continue
        week = int(t.get("week", 0) or 0)
        proposer = t.get("proposer"); receiver = t.get("receiver")
        if not proposer or not receiver:
            continue
        # wins after trade = final wins - wins up to that week
        last_week = max(team_wins_by_week.get(proposer, {0:0}).keys() or [0])
        prop_final = team_wins_by_week.get(proposer, {}).get(last_week, 0)
        prop_before = team_wins_by_week.get(proposer, {}).get(week, 0)
        recv_last_week = max(team_wins_by_week.get(receiver, {0:0}).keys() or [0])
        recv_final = team_wins_by_week.get(receiver, {}).get(recv_last_week, 0)
        recv_before = team_wins_by_week.get(receiver, {}).get(week, 0)
        labels.append(f"T{idx+1}@W{week}")
        prop_wins.append(max(0, prop_final - prop_before))
        recv_wins.append(max(0, recv_final - recv_before))

    if not labels:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.text(0.5, 0.5, "No trades with win data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(os.path.join(FIG_DIR, "23_wins_after_trade.png"))
        plt.close(fig)
        return

    x = range(len(labels)); width = 0.4
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar([i - width/2 for i in x], prop_wins, width, label="Proposer", color="#1b9e77")
    ax.bar([i + width/2 for i in x], recv_wins, width, label="Receiver", color="#d95f02")
    ax.set_title("Wins After Trade (from trade week to end)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Wins After Trade")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "23_wins_after_trade.png"))
    plt.close(fig)

def parse_start_sit_calls():
    calls = []
    for fn in os.listdir(SEASON_DIR):
        if fn.startswith("llm_calls_season_") and fn.endswith(".jsonl"):
            with open(os.path.join(SEASON_DIR, fn)) as f:
                for line in f:
                    try:
                        calls.append(json.loads(line))
                    except Exception:
                        continue
    return calls

def compute_startsit_correctness(name_to_week_points):
    calls = parse_start_sit_calls()
    per_team = defaultdict(lambda: {"total": 0, "wrong": 0, "margin_lost": 0.0})
    weekly_correctness = defaultdict(lambda: {"total":0, "correct":0})

    for c in calls:
        week = int(c.get("week", 0) or 0)
        team = c.get("team") or c.get("model") or ""
        prompt = ((c.get("input") or {}).get("prompt", ""))
        output = _normalize_output_to_dict(c.get("output"))
        if not prompt or not output:
            continue
        # Extract allowed names line
        key = "All names MUST be selected from this list:"
        idx = prompt.find(key)
        if idx == -1:
            continue
        tail = prompt[idx+len(key):]
        allowed_line = tail.split("\n", 1)[0]
        allowed = [normalize_name(x.strip()) for x in allowed_line.split(',')]

        # Starts chosen
        positions = {
            "qb": [output.get("qb")],
            "rbs": output.get("rbs", []),
            "wrs": output.get("wrs", []),
            "te": [output.get("te")],
            "flex": [output.get("flex")],
        }
        # Flatten allowed into a set
        allowed_set = set(allowed)

        # Helper to get week points
        def pts(name):
            nm = normalize_name(name)
            return float(name_to_week_points.get(nm, {}).get(week, 0.0))

        # Evaluate each slot independently vs best among allowed in that category
        slot_to_allowed = {
            "qb": [a for a in allowed if a in allowed_set],
            "rb": [a for a in allowed if a in allowed_set],
            "wr": [a for a in allowed if a in allowed_set],
            "te": [a for a in allowed if a in allowed_set],
        }
        # Simple position inference using name_to_week_points only (no position map here), so compare against all allowed
        def eval_slot(selected_names, pool_names):
            nonlocal per_team, weekly_correctness
            for sel in selected_names:
                if not sel:
                    continue
                per_team[team]["total"] += 1
                weekly_correctness[week]["total"] += 1
                sel_pts = pts(sel)
                best_pts = max([pts(nm) for nm in pool_names] + [0.0])
                if sel_pts + 1e-9 < best_pts:
                    per_team[team]["wrong"] += 1
                    per_team[team]["margin_lost"] += (best_pts - sel_pts)
                else:
                    weekly_correctness[week]["correct"] += 1

        eval_slot(positions.get("qb", []), allowed)
        eval_slot(positions.get("rbs", []), allowed)
        eval_slot(positions.get("wrs", []), allowed)
        eval_slot(positions.get("te", []), allowed)
        eval_slot(positions.get("flex", []), allowed)

    return per_team, weekly_correctness

def startsit_error_rate_by_team_graph(name_to_week_points):
    per_team, _ = compute_startsit_correctness(name_to_week_points)
    teams = list(per_team.keys()) or ["No Data"]
    rates = [ (per_team[t]["wrong"] / per_team[t]["total"])*100 if per_team[t]["total"]>0 else 0 for t in teams]
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(teams, rates, color="#C2185B")
    ax.set_title("Start/Sit Error Rate by Team")
    ax.set_ylabel("Error Rate (%)")
    ax.set_xticklabels(teams, rotation=45, ha="right")
    # Add value labels on bars
    for i, v in enumerate(rates):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "24_startsit_error_rate_by_team.png"))
    plt.close(fig)

def startsit_margin_loss_by_team_graph(name_to_week_points):
    per_team, _ = compute_startsit_correctness(name_to_week_points)
    teams = list(per_team.keys()) or ["No Data"]
    margins = [ (per_team[t]["margin_lost"] / max(1, per_team[t]["wrong"])) for t in teams]
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(teams, margins, color="#7B1FA2")
    ax.set_title("Start/Sit Avg Margin Lost on Wrong Decisions")
    ax.set_ylabel("Avg Points Lost (PPR)")
    ax.set_xticklabels(teams, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "25_startsit_margin_loss_by_team.png"))
    plt.close(fig)

def draft_projection_delta_by_team_graph(players_by_team, season_points):
    # Load canonical 2024 projections from TopPlayers_2024_Projections_PPR.csv when available
    proj_map = read_top_projections()
    teams = list(players_by_team.keys()) or ["No Data"]
    deltas = []
    for t in teams:
        total = 0.0
        for p in players_by_team.get(t, []):
            nm = normalize_name(p.get("name"))
            # Prefer canonical projection; fall back to csv value
            proj = float(proj_map.get(nm, p.get("proj") or 0) or 0)
            actual = float(season_points.get(nm, 0.0))
            total += (actual - proj)
        deltas.append(total)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(teams, deltas, color="#2E7D32")
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_title("Draft Projection Delta by Team (Actual - Projected)")
    ax.set_ylabel("Total Delta (PPR)")
    ax.set_xticklabels(teams, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "26_draft_projection_delta_by_team.png"))
    plt.close(fig)

def draft_projection_delta_by_position_box(players_by_team, season_points):
    pos_to_deltas = defaultdict(list)
    proj_map = read_top_projections()
    for t, roster in players_by_team.items():
        for p in roster:
            nm = normalize_name(p.get("name"))
            pos = p.get("position") or "Other"
            proj = float(proj_map.get(nm, p.get("proj") or 0) or 0)
            actual = float(season_points.get(nm, 0.0))
            pos_to_deltas[pos].append(actual - proj)
    positions = sorted(pos_to_deltas.keys()) or ["No Data"]
    data = [pos_to_deltas[pos] or [0.0] for pos in positions]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.boxplot(data, labels=positions)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_title("Draft Projection Delta by Position")
    ax.set_ylabel("Actual - Projected (PPR)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "27_draft_projection_delta_by_position.png"))
    plt.close(fig)

def draft_hit_rate_by_team_graph(players_by_team, season_points):
    proj_map = read_top_projections()
    teams = list(players_by_team.keys()) or ["No Data"]
    rates = []
    for t in teams:
        hits = 0; total = 0
        for p in players_by_team.get(t, []):
            nm = normalize_name(p.get("name"))
            proj = float(proj_map.get(nm, p.get("proj") or 0) or 0)
            actual = float(season_points.get(nm, 0.0))
            total += 1
            if actual >= proj:
                hits += 1
        rates.append((hits / total)*100 if total else 0)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(teams, rates, color="#1976D2")
    ax.set_title("Draft Hit Rate by Team (Actual >= Projected)")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_xticklabels(teams, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "28_draft_hit_rate_by_team.png"))
    plt.close(fig)

def trade_activity_vs_value_graph(trades, name_to_week_points):
    per_team, _ = compute_trade_values(trades, name_to_week_points)
    activity = Counter()
    for t in trades:
        if not isinstance(t, dict):
            continue
        if t.get("proposer"):
            activity[t["proposer"]] += 1
        if t.get("receiver"):
            activity[t["receiver"]] += 1
    teams = sorted(set(per_team.keys()) | set(activity.keys())) or ["No Data"]
    xs = [activity.get(t, 0) for t in teams]
    ys = [per_team.get(t, 0.0) for t in teams]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(xs, ys, s=100, alpha=0.7)
    for i, t in enumerate(teams):
        ax.annotate(t, (xs[i], ys[i]), fontsize=8)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_title("Trade Aggressiveness vs Value Gained")
    ax.set_xlabel("Trades Participated")
    ax.set_ylabel("Cumulative Trade Value (PPR)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "29_trade_aggressiveness_vs_value.png"))
    plt.close(fig)

def startsit_correctness_over_time_graph(name_to_week_points):
    _, weekly_corr = compute_startsit_correctness(name_to_week_points)
    weeks = sorted(weekly_corr.keys()) or [0]
    rates = [ (weekly_corr[w]["correct"] / weekly_corr[w]["total"])*100 if weekly_corr[w]["total"]>0 else 0 for w in weeks]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(weeks, rates, marker='o')
    ax.set_title("Start/Sit Correctness Over Time")
    ax.set_xlabel("Week")
    ax.set_ylabel("Correctness Rate (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "30_startsit_correctness_over_time.png"))
    plt.close(fig)

def startsit_correctness_matrix_by_team_week(name_to_week_points):
    calls = parse_start_sit_calls()
    # team -> week -> {total_slots, correct_slots}
    mat = defaultdict(lambda: defaultdict(lambda: {"total":0, "correct":0}))

    for c in calls:
        week = int(c.get("week", 0) or 0)
        team = c.get("team") or c.get("model") or ""
        prompt = ((c.get("input") or {}).get("prompt", ""))
        output = _normalize_output_to_dict(c.get("output"))
        if not prompt or not output:
            continue
        key = "All names MUST be selected from this list:"
        idx = prompt.find(key)
        if idx == -1:
            continue
        tail = prompt[idx+len(key):]
        allowed_line = tail.split("\n", 1)[0]
        allowed = [normalize_name(x.strip()) for x in allowed_line.split(',')]

        def pts(name):
            nm = normalize_name(name)
            return float(name_to_week_points.get(nm, {}).get(week, 0.0))

        # build selected list
        selected = []
        for k in ("qb",):
            v = output.get(k)
            if v:
                selected.append(v)
        for k in ("rbs","wrs"):
            for v in output.get(k, []) or []:
                if v:
                    selected.append(v)
        for k in ("te","flex"):
            v = output.get(k)
            if v:
                selected.append(v)

        # For each selected player, correct if it equals the top scoring among allowed
        if allowed:
            best_name = None
            best_pts = -1e9
            for nm in allowed:
                p = pts(nm)
                if p > best_pts:
                    best_pts = p
                    best_name = nm
            for sel in selected:
                mat[team][week]["total"] += 1
                if normalize_name(sel) == best_name:
                    mat[team][week]["correct"] += 1

    # Plot: grouped bars per week, colored by team, height = correctness rate (%)
    teams = sorted(mat.keys())
    weeks = sorted({w for team_weeks in mat.values() for w in team_weeks.keys()})
    if not teams or not weeks:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.text(0.5, 0.5, "No start/sit correctness data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(os.path.join(FIG_DIR, "31_startsit_correctness_by_team_week.png"))
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    width = 0.8 / max(1, len(teams))
    x_ticks = list(range(len(weeks)))
    for i, team in enumerate(teams):
        vals = []
        for w in weeks:
            tot = mat[team][w]["total"]
            corr = mat[team][w]["correct"]
            rate = (corr / tot)*100 if tot else 0
            vals.append(rate)
        xs = [x + i*width - 0.4 + width/2 for x in x_ticks]
        ax.bar(xs, vals, width=width, label=team)

    ax.set_title("Start/Sit Correctness by Week and Team (Top-Scoring Pick %)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Correctness Rate (%)")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(w) for w in weeks])
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "31_startsit_correctness_by_team_week.png"))
    plt.close(fig)

def main():
    standings = read_final_standings()
    weekly = read_weekly_results()
    trades = read_accepted_trades()
    trade_calls = read_trade_calls()
    draft = read_draft_team_csvs()

    # Original 10 graphs
    bar_total_points_vs_wins(standings)
    line_cumulative_points(weekly)
    line_cumulative_wins(weekly)
    box_weekly_points_distribution(weekly)
    draft_position_breakdown(draft)
    draft_projected_points(draft)
    trade_volume_by_week(trade_calls)
    accepted_trades_by_team(trades)
    model_performance_scatter(weekly)
    head_to_head_heatmap(weekly)

    # New 10 deeper insight graphs
    trade_negotiation_patterns(trade_calls)
    draft_strategy_value_vs_projections(draft)
    start_sit_recent_vs_historical()
    trade_acceptance_rates_by_model(trade_calls)
    player_selection_patterns_by_model(draft)
    negotiation_complexity_analysis(trade_calls)
    strategic_trade_value_analysis(trades)
    model_specific_behavior_patterns(weekly)
    decision_consistency_over_time(trade_calls)
    performance_correlation_with_trading_activity(weekly, trade_calls)

    # Correctness graphs
    name_to_week_points, name_to_position = read_player_week_points()
    season_points = read_player_season_points()
    trade_value_by_team_graph(trades, name_to_week_points)
    trade_value_per_trade_graph(trades, name_to_week_points)
    wins_after_trade_graph(weekly, trades)
    startsit_error_rate_by_team_graph(name_to_week_points)
    startsit_margin_loss_by_team_graph(name_to_week_points)
    draft_projection_delta_by_team_graph(draft, season_points)
    draft_projection_delta_by_position_box(draft, season_points)
    draft_hit_rate_by_team_graph(draft, season_points)
    trade_activity_vs_value_graph(trades, name_to_week_points)
    startsit_correctness_over_time_graph(name_to_week_points)
    startsit_correctness_matrix_by_team_week(name_to_week_points)
    startsit_correctness_matrix_by_team_week(name_to_week_points)

if __name__ == "__main__":
    main()


