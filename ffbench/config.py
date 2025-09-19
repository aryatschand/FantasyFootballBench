import os
import json
from functools import lru_cache


def _default_config_path():
    # Project root
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "config.json")


@lru_cache(maxsize=1)
def get_config():
    path = os.environ.get("FFBENCH_CONFIG", _default_config_path())
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}. Set FFBENCH_CONFIG or create config.json at project root.")
    with open(path) as f:
        cfg = json.load(f)
    # Basic validation
    if "num_teams" not in cfg or "models" not in cfg or "roster_slots" not in cfg or "scoring" not in cfg:
        raise ValueError("Config missing required keys: num_teams, models, roster_slots, scoring")
    return cfg


def get_scoring():
    return get_config()["scoring"]


def get_roster_slots():
    # Returns list of (slot, count)
    slots = get_config()["roster_slots"]
    return [(s["slot"], int(s["count"])) for s in slots]


def get_models():
    cfg = get_config()
    raw = cfg["models"]
    n = int(cfg["num_teams"])
    if not raw:
        raise ValueError("Config models list is empty")
    # Normalize to list of {id, name}
    models = []
    for item in raw:
        if isinstance(item, str):
            models.append({"id": item, "name": item})
        elif isinstance(item, dict):
            mid = item.get("id") or item.get("arn")
            if mid:
                models.append({"id": mid, "name": item.get("name", mid)})
    # Repeat to match team count
    if len(models) < n:
        times = (n + len(models) - 1) // len(models)
        models = (models * times)[:n]
    return models[:n]


def get_model_ids():
    return [m["id"] for m in get_models()]


def get_num_teams():
    return int(get_config()["num_teams"])


def format_roster_format():
    """Format roster format for prompts"""
    slots = get_roster_slots()
    formatted = ", ".join([f"{count} {slot}" for slot, count in slots])
    return f"Roster Format: {formatted}"


def format_scoring_format():
    """Format scoring format for prompts"""
    scoring = get_scoring()
    formatted_parts = []
    for stat, points in scoring.items():
        if stat == "passing_yards":
            formatted_parts.append(f"Passing Yards ({points} pts/yard)")
        elif stat == "passing_tds":
            formatted_parts.append(f"Passing TDs ({points} pts)")
        elif stat == "interceptions":
            formatted_parts.append(f"Interceptions ({points} pts)")
        elif stat == "rushing_yards":
            formatted_parts.append(f"Rushing Yards ({points} pts/yard)")
        elif stat == "rushing_tds":
            formatted_parts.append(f"Rushing TDs ({points} pts)")
        elif stat == "receptions":
            formatted_parts.append(f"Receptions ({points} pt)")
        elif stat == "receiving_yards":
            formatted_parts.append(f"Receiving Yards ({points} pts/yard)")
        elif stat == "receiving_tds":
            formatted_parts.append(f"Receiving TDs ({points} pts)")
        elif stat == "fumbles_lost":
            formatted_parts.append(f"Fumbles Lost ({points} pts)")
    return f"Scoring Format: {', '.join(formatted_parts)}"


