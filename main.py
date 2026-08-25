#!/usr/bin/env python3
"""FantasyFootballBench entry point.

Run `python main.py --help` for the available phases. Each subcommand maps to a
script under scripts/; `all` runs the same sequence as scripts/run_full_simulation.sh.
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ffbench.env import load_env

ROOT = os.path.dirname(os.path.abspath(__file__))


def _ensure_sim_id() -> str:
    """Create (or reuse) the simulation id that every phase writes under."""
    sim_id = os.environ.get("FFBENCH_SIM_ID")
    if not sim_id:
        sim_id = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.environ["FFBENCH_SIM_ID"] = sim_id
    sim_meta_dir = os.path.join(ROOT, "data", "simulations")
    os.makedirs(sim_meta_dir, exist_ok=True)
    with open(os.path.join(sim_meta_dir, "latest_simulation_id.txt"), "w") as f:
        f.write(sim_id)
    return sim_id


def _require_api_key() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        sys.exit(
            "ERROR: OPENROUTER_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key, or export it in your shell."
        )


def cmd_check(args):
    _require_api_key()
    from scripts.test_models_from_config import main as run_check
    run_check()


def cmd_export(args):
    from scripts.export_top_players import export_top_players_2024
    export_top_players_2024()


def cmd_draft(args):
    _require_api_key()
    sim_id = _ensure_sim_id()
    print(f"Simulation ID: {sim_id}")
    from scripts.run_draft import complete_draft_quickly, draft_simulation, resume_draft
    if args.resume:
        resume_draft(args.simulation_id)
    elif args.complete:
        complete_draft_quickly(args.simulation_id)
    else:
        draft_simulation()


def cmd_season(args):
    _require_api_key()
    sim_id = _ensure_sim_id()
    print(f"Simulation ID: {sim_id}")
    from scripts.simulate_season import simulate_season
    simulate_season(start_week=args.start_week)


def cmd_figures(args):
    from scripts.generate_blog_figures import main as run_figures
    run_figures()


def cmd_demo(args):
    from scripts.demo_data_access import main as run_demo
    run_demo()


def cmd_all(args):
    _require_api_key()
    sim_id = _ensure_sim_id()
    print(f"Simulation ID: {sim_id}\n")
    print("[1/4] Checking model connectivity from config.json...")
    try:
        cmd_check(args)
    except Exception as exc:  # connectivity is informational, not fatal
        print(f"  (connectivity check failed: {exc})")
    print("\n[2/4] Exporting projection-based player rankings...")
    cmd_export(args)
    print("\n[3/4] Running draft simulation...")
    cmd_draft(args)
    print("\n[4/4] Simulating the full season...")
    cmd_season(args)
    out = os.path.join(ROOT, "data", "simulations", sim_id)
    print(f"\nDone. Outputs saved under {out}/")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="FantasyFootballBench — LLM benchmark over a simulated fantasy football season.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Verify every model in config.json responds").set_defaults(func=cmd_check)
    sub.add_parser("export", help="Export projection-based player rankings").set_defaults(func=cmd_export)

    p_draft = sub.add_parser("draft", help="Run the draft phase")
    p_draft.add_argument("--resume", action="store_true", help="Resume an interrupted draft")
    p_draft.add_argument("--complete", action="store_true", help="Fill remaining picks without LLM calls")
    p_draft.add_argument("--simulation-id", dest="simulation_id", default=None)
    p_draft.set_defaults(func=cmd_draft)

    p_season = sub.add_parser("season", help="Run the season phase (needs an existing draft)")
    p_season.add_argument("--start-week", type=int, default=1)
    p_season.set_defaults(func=cmd_season)

    sub.add_parser("figures", help="Regenerate analysis figures").set_defaults(func=cmd_figures)
    sub.add_parser("demo", help="Print sample DataHandler queries (no API calls)").set_defaults(func=cmd_demo)

    p_all = sub.add_parser("all", help="Run the full pipeline: check, export, draft, season")
    p_all.add_argument("--start-week", type=int, default=1)
    p_all.add_argument("--resume", action="store_true")
    p_all.add_argument("--complete", action="store_true")
    p_all.add_argument("--simulation-id", dest="simulation_id", default=None)
    p_all.set_defaults(func=cmd_all)

    return parser


def main():
    load_env()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
