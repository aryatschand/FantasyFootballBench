#!/usr/bin/env python3
"""
Generate blog figures for FantasyFootballBench research paper.
Creates 5 professional visualizations from simulation data.
"""

import os
import sys
import csv
import json
import glob
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for generating images
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw, ImageFont
import imageio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).parent.parent
SIMULATIONS_DIR = ROOT / "data" / "simulations"
OUTPUT_DIR = ROOT / "figures"
ICONS_DIR = ROOT / "icons"
OUTPUT_DIR.mkdir(exist_ok=True)

# All simulation folders to aggregate
SIMULATION_IDS = [
    "simulation_20250919_152504_INJURY",
    "simulation_20251015_224441_INJURY",
    "simulation_20251016_114217_FIXED",
    "simulation_20251016_181800_FIXED",
    "simulation_20251020_174956",
]

# Simulation labels for videos
SIMULATION_LABELS = ["A", "B", "C", "D", "E"]

# Model display names, colors, and icon files
MODEL_INFO = {
    "openai/gpt-5": {"name": "GPT-5", "color": "#10a37f", "short": "GPT5", "icon": "gpt.png"},
    "openai/gpt-5-mini": {"name": "GPT-5 Mini", "color": "#74aa9c", "short": "GPT5m", "icon": "gpt.png"},
    "openai/gpt-oss-120b": {"name": "GPT-OSS", "color": "#1a7f64", "short": "GPT-O", "icon": "gpt.png"},
    "anthropic/claude-sonnet-4": {"name": "Claude Sonnet 4", "color": "#cc785c", "short": "CSon4", "icon": "claude.png"},
    "anthropic/claude-opus-4.1": {"name": "Claude Opus 4.1", "color": "#d4a574", "short": "COp4", "icon": "claude.png"},
    "google/gemini-2.5-pro": {"name": "Gemini 2.5 Pro", "color": "#4285f4", "short": "Gem2P", "icon": "gemini.png"},
    "google/gemini-2.5-flash": {"name": "Gemini 2.5 Flash", "color": "#669df6", "short": "Gem2F", "icon": "gemini.png"},
    "qwen/qwen3-max": {"name": "Qwen 3 Max", "color": "#6366f1", "short": "Qwen3", "icon": "qwen.png"},
    "moonshotai/kimi-k2-0905": {"name": "Kimi K2", "color": "#8b5cf6", "short": "Kimi", "icon": "kimi.png"},
    "meta-llama/llama-4-maverick": {"name": "Llama 4 Maverick", "color": "#0668E1", "short": "Llama4", "icon": "llama.png"},
}

def get_icon_for_model_name(model_name):
    """Determine the correct icon based on the model name string."""
    model_name_lower = model_name.lower()
    if "claude" in model_name_lower:
        return "claude.png"
    elif "gemini" in model_name_lower:
        return "gemini.png"
    elif "gpt" in model_name_lower:
        return "gpt.png"
    elif "kimi" in model_name_lower:
        return "kimi.png"
    elif "qwen" in model_name_lower:
        return "qwen.png"
    elif "llama" in model_name_lower:
        return "llama.png"
    else:
        return "gpt.png"  # Default fallback

# Rank colors (gold, silver, bronze)
RANK_COLORS = {
    1: "#FFD700",  # Gold
    2: "#C0C0C0",  # Silver
    3: "#CD7F32",  # Bronze
}

# Cache for loaded icons
_icon_cache = {}

# Team name to model mapping (extracted from simulations)
def get_team_model_mapping(sim_dir):
    """Extract team name -> model ID mapping from draft CSVs."""
    mapping = {}
    draft_dir = sim_dir / "draft_results"
    for csv_file in draft_dir.glob("*_2024.csv"):
        if "TopPlayers" in csv_file.name:
            continue
        team_name = csv_file.stem.replace("_2024", "")
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_id = row.get("Model", "")
                if model_id:
                    mapping[team_name] = model_id
                    break
    return mapping

# Position colors for stacked bar
POSITION_COLORS = {
    "QB": "#e74c3c",
    "RB": "#3498db",
    "WR": "#2ecc71",
    "TE": "#f39c12",
}

# =============================================================================
# Utility Functions
# =============================================================================

def load_weekly_results(sim_dir, week):
    """Load weekly results CSV and return DataFrame."""
    path = sim_dir / "season_results" / f"week_{week}_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def load_all_weekly_results(sim_dir):
    """Load all weekly results for a simulation."""
    all_weeks = {}
    for week in range(1, 18):
        df = load_weekly_results(sim_dir, week)
        if not df.empty:
            all_weeks[week] = df
    return all_weeks

def load_draft_rosters(sim_dir):
    """Load all team draft rosters."""
    rosters = {}
    draft_dir = sim_dir / "draft_results"
    for csv_file in draft_dir.glob("*_2024.csv"):
        if "TopPlayers" in csv_file.name:
            continue
        team_name = csv_file.stem.replace("_2024", "")
        rosters[team_name] = pd.read_csv(csv_file)
    return rosters

def load_projections(sim_dir):
    """Load projection data."""
    path = sim_dir / "draft_results" / "TopPlayers_2024_Projections_PPR.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def load_accepted_trades(sim_dir):
    """Load accepted trades from JSONL."""
    trades = []
    trade_files = list((sim_dir / "season_results").glob("accepted_trades_*.jsonl"))
    for tf in trade_files:
        with open(tf) as f:
            for line in f:
                try:
                    trades.append(json.loads(line.strip()))
                except:
                    continue
    return trades

def load_2023_stats():
    """Load 2023 season stats for past year performance."""
    path = ROOT / "data" / "2023" / "PlayerSeasonStats.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def load_2024_stats():
    """Load 2024 season stats for actual performance."""
    path = ROOT / "data" / "2024" / "PlayerSeasonStats.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def calculate_ppr_points(row):
    """Calculate PPR fantasy points from stats row."""
    pts = 0
    pts += float(row.get("PassingYards", 0) or 0) * 0.04
    pts += float(row.get("PassingTouchdowns", 0) or 0) * 4
    pts += float(row.get("PassingInterceptions", 0) or 0) * -2
    pts += float(row.get("RushingYards", 0) or 0) * 0.1
    pts += float(row.get("RushingTouchdowns", 0) or 0) * 6
    pts += float(row.get("Receptions", 0) or 0) * 1
    pts += float(row.get("ReceivingYards", 0) or 0) * 0.1
    pts += float(row.get("ReceivingTouchdowns", 0) or 0) * 6
    pts += float(row.get("FumblesLost", 0) or 0) * -2
    return pts

def get_model_display_name(model_id):
    """Get display name for model."""
    return MODEL_INFO.get(model_id, {}).get("name", model_id.split("/")[-1])

def get_model_color(model_id):
    """Get color for model."""
    return MODEL_INFO.get(model_id, {}).get("color", "#888888")

def get_model_icon(model_id_or_name, size=30, fix_aspect=True):
    """Load and return model icon as a PIL Image, cached and resized.
    
    Can accept either a model_id (like 'openai/gpt-5') or a team/model name 
    (like 'GPT 5' or 'Claude Sonnet 4').
    
    fix_aspect: If True, apply aspect ratio fixes for problematic logos (like Llama)
    """
    cache_key = (model_id_or_name, size, fix_aspect)
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]
    
    # First try to get icon from MODEL_INFO by model_id
    icon_file = MODEL_INFO.get(model_id_or_name, {}).get("icon")
    
    # If not found, try to determine icon from the name string
    if not icon_file:
        icon_file = get_icon_for_model_name(model_id_or_name)
    
    icon_path = ICONS_DIR / icon_file
    
    if not icon_path.exists():
        # Return None if icon doesn't exist
        return None
    
    try:
        img = Image.open(icon_path).convert("RGBA")
        
        # Special handling for Llama logo - squish vertically to fix aspect ratio
        if fix_aspect and "llama" in icon_file.lower():
            # Squish vertically (make it wider relative to height)
            new_width = size
            new_height = int(size * 0.75)  # 75% of height to squish
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            # Resize to square
            img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        _icon_cache[cache_key] = img
        return img
    except Exception as e:
        print(f"  Warning: Could not load icon {icon_path}: {e}")
        return None

def get_icon_zoom_multiplier(model_name):
    """Get zoom multiplier for specific model icons that need size adjustment."""
    model_name_lower = model_name.lower()
    if "gpt" in model_name_lower:
        return 0.95  # GPT icons slightly smaller (was too big)
    return 0.95  # All icons 5% smaller

def add_icon_to_axis(ax, model_id_or_name, x, y, size=0.04, transform=None):
    """Add a model icon to the axis at specified position."""
    icon = get_model_icon(model_id_or_name, size=80)  # Larger base size
    if icon is None:
        return
    
    # Convert PIL to numpy array
    icon_array = np.array(icon)
    
    # Apply zoom multiplier (all icons slightly smaller now)
    zoom_mult = get_icon_zoom_multiplier(model_id_or_name)
    
    imagebox = OffsetImage(icon_array, zoom=size * 10 * zoom_mult)  # Reduced from 12 to 10
    if transform is None:
        transform = ax.transAxes
    ab = AnnotationBbox(imagebox, (x, y), frameon=False, 
                        xycoords=transform, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)

def normalize_team_name(team_name):
    """Normalize team name by extracting the model name part."""
    # Remove _Team_X suffix
    parts = team_name.rsplit("_Team_", 1)
    return parts[0].replace("_", " ") if parts else team_name

# =============================================================================
# Figure 1: Weekly Standings Video (for all simulations)
# =============================================================================

def generate_weekly_standings_video_for_sim(sim_id, sim_label):
    """Generate a video showing standings progression week by week for one simulation."""
    
    sim_dir = SIMULATIONS_DIR / sim_id
    weekly_results = load_all_weekly_results(sim_dir)
    team_model_map = get_team_model_mapping(sim_dir)
    trades = load_accepted_trades(sim_dir)
    
    if not weekly_results:
        print(f"  No weekly results found for {sim_id}!")
        return
    
    # Count trades per team through each week
    trades_by_team_by_week = defaultdict(lambda: defaultdict(int))
    for trade in trades:
        week = trade.get("week", 1)
        proposer = trade.get("proposer", "")
        receiver = trade.get("receiver", "")
        # Trades count up through that week
        for w in range(week, 18):
            trades_by_team_by_week[proposer][w] += 1
            trades_by_team_by_week[receiver][w] += 1
    
    # Calculate cumulative standings through each week
    standings_by_week = {}
    cumulative = defaultdict(lambda: {"wins": 0, "losses": 0, "points_for": 0.0, "points_against": 0.0, "top_player": "", "top_score": 0.0, "player_scores": defaultdict(float), "trades": 0})
    
    for week in range(1, 18):
        if week not in weekly_results:
            continue
        
        df = weekly_results[week]
        
        # Get matchups (TOTAL rows)
        totals = df[df["Player"] == "TOTAL"]
        
        # Process each matchup
        processed = set()
        for _, row in totals.iterrows():
            team = row["Team"]
            opponent = row["Opponent"]
            points = float(row["Points"])
            
            if (team, opponent) in processed or (opponent, team) in processed:
                continue
            
            # Find opponent's points
            opp_row = totals[(totals["Team"] == opponent) & (totals["Opponent"] == team)]
            if opp_row.empty:
                continue
            
            opp_points = float(opp_row.iloc[0]["Points"])
            
            # Update standings
            cumulative[team]["points_for"] += points
            cumulative[team]["points_against"] += opp_points
            cumulative[opponent]["points_for"] += opp_points
            cumulative[opponent]["points_against"] += points
            
            if points > opp_points:
                cumulative[team]["wins"] += 1
                cumulative[opponent]["losses"] += 1
            elif opp_points > points:
                cumulative[opponent]["wins"] += 1
                cumulative[team]["losses"] += 1
            
            processed.add((team, opponent))
        
        # Track individual player scores for top player
        player_rows = df[(df["Player"] != "TOTAL") & (df["Position"] != "")]
        for _, row in player_rows.iterrows():
            team = row["Team"]
            player = row["Player"]
            pts = float(row["Points"])
            cumulative[team]["player_scores"][player] += pts
        
        # Find top player and update trades for each team
        for team in cumulative:
            scores = cumulative[team]["player_scores"]
            if scores:
                top_player = max(scores.keys(), key=lambda p: scores[p])
                cumulative[team]["top_player"] = top_player
                cumulative[team]["top_score"] = scores[top_player]
            cumulative[team]["trades"] = trades_by_team_by_week[team][week]
        
        # Store snapshot
        standings_by_week[week] = {
            team: dict(data) for team, data in cumulative.items()
        }
    
    # Generate frames
    frames = []
    
    for week in range(1, 18):
        if week not in standings_by_week:
            continue
        
        standings = standings_by_week[week]
        
        # Sort by wins, then points
        sorted_teams = sorted(
            standings.items(),
            key=lambda x: (x[1]["wins"], x[1]["points_for"]),
            reverse=True
        )
        
        # Create frame - white background
        fig, ax = plt.subplots(figsize=(12, 9), facecolor='#ffffff')
        ax.set_facecolor('#ffffff')
        
        # Title - more compact
        ax.text(0.5, 0.96, f"FantasyFootballBench Standings", fontsize=26, fontweight='bold',
                color='#1a1a2e', ha='center', transform=ax.transAxes, fontfamily='sans-serif')
        ax.text(0.5, 0.90, f"Simulation {sim_label} — Week {week}", fontsize=18, 
                color='#555555', ha='center', transform=ax.transAxes)
        
        # Headers - center aligned, top scorer thinner
        headers = ["#", "", "Team", "W", "L", "PF", "PA", "Trades", "Top Scorer"]
        x_positions = [0.06, 0.11, 0.17, 0.38, 0.45, 0.52, 0.61, 0.70, 0.80]
        
        for i, (header, x) in enumerate(zip(headers, x_positions)):
            ax.text(x, 0.84, header, fontsize=11, fontweight='bold',
                   color='#333333', transform=ax.transAxes)
        
        # Draw line under headers
        ax.plot([0.02, 0.98], [0.82, 0.82], color='#333366', linewidth=2, transform=ax.transAxes, clip_on=False)
        
        # Team rows
        y_start = 0.77
        row_height = 0.07
        
        for rank, (team_name, data) in enumerate(sorted_teams, 1):
            y = y_start - (rank - 1) * row_height
            
            # Get model info
            model_id = team_model_map.get(team_name, "")
            model_name = normalize_team_name(team_name)
            
            # Rank with gold/silver/bronze coloring (default dark gray for white bg)
            rank_color = RANK_COLORS.get(rank, '#333333')
            ax.text(x_positions[0], y, f"{rank}", fontsize=14, color=rank_color,
                   transform=ax.transAxes, fontweight='bold')
            
            # Model icon - use model_name for matching, moved right and slightly up
            add_icon_to_axis(ax, model_name, x_positions[1] + 0.02, y + 0.005, size=0.035, transform=ax.transAxes)
            
            # Team name
            ax.text(x_positions[2], y, model_name, fontsize=12, color='#1a1a2e',
                   transform=ax.transAxes, fontweight='bold')
            
            # Stats - colors adjusted for white background
            ax.text(x_positions[3], y, str(data["wins"]), fontsize=13, color='#27ae60',
                   transform=ax.transAxes, fontweight='bold')
            ax.text(x_positions[4], y, str(data["losses"]), fontsize=13, color='#c0392b',
                   transform=ax.transAxes, fontweight='bold')
            ax.text(x_positions[5], y, f"{data['points_for']:.0f}", fontsize=12, color='#2980b9',
                   transform=ax.transAxes)
            ax.text(x_positions[6], y, f"{data['points_against']:.0f}", fontsize=12, color='#d35400',
                   transform=ax.transAxes)
            ax.text(x_positions[7], y, str(data.get("trades", 0)), fontsize=12, color='#8e44ad',
                   transform=ax.transAxes, fontweight='bold')
            
            # Top scorer - balanced width
            top_text = f"{data['top_player'][:14]}..." if len(data['top_player']) > 14 else data['top_player']
            ax.text(x_positions[8], y, f"{top_text} ({data['top_score']:.0f})", 
                   fontsize=10, color='#b7950b', transform=ax.transAxes)
        
        ax.axis('off')
        
        # Tighter margins
        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        
        # Save frame to buffer
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        frame = frame[:, :, :3]
        frames.append(frame.copy())
        plt.close(fig)
    
    # Create video (1 second per frame)
    output_path = OUTPUT_DIR / f"weekly_standings_{sim_label}.mp4"
    imageio.mimwrite(str(output_path), frames, fps=1, format='FFMPEG')
    print(f"  Saved: {output_path}")
    
    # Also save as GIF
    gif_path = OUTPUT_DIR / f"weekly_standings_{sim_label}.gif"
    imageio.mimwrite(str(gif_path), frames, duration=1000, loop=0, format='GIF')
    print(f"  Saved: {gif_path}")


def generate_weekly_standings_video():
    """Generate videos for all simulations."""
    print("Generating Figure 1: Weekly Standings Videos (all simulations)...")
    
    for i, sim_id in enumerate(SIMULATION_IDS):
        label = SIMULATION_LABELS[i]
        print(f"  Processing Simulation {label} ({sim_id})...")
        generate_weekly_standings_video_for_sim(sim_id, label)

# =============================================================================
# Figure 2: Point Distributions by Position (Stacked Bar)
# =============================================================================

def generate_position_distribution():
    """Generate stacked bar chart of points by position for each team."""
    print("Generating Figure 2: Position Point Distribution...")
    
    # Aggregate across all simulations
    team_position_points = defaultdict(lambda: defaultdict(list))  # team -> position -> [points per sim]
    team_models = {}
    
    for sim_id in SIMULATION_IDS:
        sim_dir = SIMULATIONS_DIR / sim_id
        if not sim_dir.exists():
            continue
        
        weekly_results = load_all_weekly_results(sim_dir)
        team_model_map = get_team_model_mapping(sim_dir)
        
        # Track points per team per position for this sim
        sim_points = defaultdict(lambda: defaultdict(float))
        
        for week, df in weekly_results.items():
            player_rows = df[(df["Player"] != "TOTAL") & (df["Position"] != "")]
            for _, row in player_rows.iterrows():
                team = normalize_team_name(row["Team"])
                position = row["Position"]
                pts = float(row["Points"])
                sim_points[team][position] += pts
                
                # Track model
                if team not in team_models:
                    model_id = team_model_map.get(row["Team"], "")
                    team_models[team] = model_id
        
        # Store this simulation's totals
        for team, positions in sim_points.items():
            for pos, pts in positions.items():
                team_position_points[team][pos].append(pts)
    
    # Calculate averages
    team_avg_points = {}
    for team, positions in team_position_points.items():
        team_avg_points[team] = {pos: np.mean(pts_list) for pos, pts_list in positions.items()}
    
    # Sort teams by total points
    teams_sorted = sorted(
        team_avg_points.keys(),
        key=lambda t: sum(team_avg_points[t].values()),
        reverse=True
    )
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(18, 10), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')
    
    x = np.arange(len(teams_sorted))
    width = 0.7
    
    # Stack positions
    bottom = np.zeros(len(teams_sorted))
    positions_order = ["QB", "RB", "WR", "TE"]
    
    bars_by_position = {}
    for pos in positions_order:
        values = [team_avg_points[team].get(pos, 0) for team in teams_sorted]
        bars = ax.bar(x, values, width, bottom=bottom, label=pos, 
                     color=POSITION_COLORS[pos], edgecolor='white', linewidth=0.5)
        bars_by_position[pos] = bars
        bottom += np.array(values)
    
    # Add total on top of each bar
    for i, team in enumerate(teams_sorted):
        total = sum(team_avg_points[team].values())
        ax.text(i, total + 30, f"{total:.0f}", ha='center', va='bottom',
               fontsize=14, fontweight='bold', color='#333333')
    
    # Formatting - BIGGER FONTS
    ax.set_xlabel("Team (Model)", fontsize=18, fontweight='bold', labelpad=55)
    ax.set_ylabel("Average Fantasy Points Per Season", fontsize=18, fontweight='bold')
    ax.set_title("Fantasy Points Distribution by Position\n(Averaged Across 5 Simulations)", 
                fontsize=24, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(" ", "\n") for t in teams_sorted], fontsize=13, rotation=0)
    ax.set_ylim(0, max(bottom) * 1.12)
    
    # Add model icons BELOW the plot - use blended transform (data x, axes y)
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, team in enumerate(teams_sorted):
        icon = get_model_icon(team, size=70)
        if icon is not None:
            icon_array = np.array(icon)
            zoom_mult = get_icon_zoom_multiplier(team)
            imagebox = OffsetImage(icon_array, zoom=0.55 * zoom_mult)
            # x position is data coordinate (matches tick), y is axes fraction (below plot) - moved down slightly
            ab = AnnotationBbox(imagebox, (i, -0.14), frameon=False,
                               xycoords=trans, box_alignment=(0.5, 0.5),
                               annotation_clip=False)
            ax.add_artist(ab)
    ax.tick_params(axis='y', labelsize=14)
    
    # Legend
    legend = ax.legend(loc='upper right', fontsize=14, framealpha=0.95)
    legend.set_title("Position", prop={'weight': 'bold', 'size': 14})
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "position_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"  Saved: {output_path}")

# =============================================================================
# Figure 3: Draft Performance (Past Year vs Projected vs Actual)
# =============================================================================

def generate_draft_performance():
    """Generate grouped bar chart comparing draft strategies."""
    print("Generating Figure 3: Draft Performance Comparison...")
    
    # Load historical and actual stats
    stats_2023 = load_2023_stats()
    stats_2024 = load_2024_stats()
    
    # Aggregate across simulations
    team_past = defaultdict(list)       # 2023 points of drafted players
    team_projected = defaultdict(list)  # Projected 2024 points
    team_actual = defaultdict(list)     # Actual 2024 points
    team_models = {}
    
    for sim_id in SIMULATION_IDS:
        sim_dir = SIMULATIONS_DIR / sim_id
        if not sim_dir.exists():
            continue
        
        rosters = load_draft_rosters(sim_dir)
        projections = load_projections(sim_dir)
        team_model_map = get_team_model_mapping(sim_dir)
        
        for team_name, roster_df in rosters.items():
            team_display = normalize_team_name(team_name)
            
            if team_display not in team_models:
                team_models[team_display] = team_model_map.get(team_name, "")
            
            sim_past = 0.0
            sim_projected = 0.0
            sim_actual = 0.0
            
            for _, player in roster_df.iterrows():
                player_name = player["Name"]
                
                # 2023 stats (past year)
                if not stats_2023.empty:
                    match_2023 = stats_2023[stats_2023["Name"].str.contains(player_name.split()[-1], case=False, na=False)]
                    if not match_2023.empty:
                        # Use FantasyPointsPPR if available
                        if "FantasyPointsPPR" in match_2023.columns:
                            sim_past += float(match_2023.iloc[0].get("FantasyPointsPPR", 0) or 0)
                        else:
                            sim_past += calculate_ppr_points(match_2023.iloc[0])
                
                # Projected points
                if not projections.empty:
                    proj_match = projections[projections["Name"] == player_name]
                    if not proj_match.empty:
                        sim_projected += float(proj_match.iloc[0].get("FantasyPoints", 0) or 0)
                
                # Actual 2024 points (from draft CSV, which has actual season points)
                sim_actual += float(player.get("FantasyPoints", 0) or 0)
            
            team_past[team_display].append(sim_past)
            team_projected[team_display].append(sim_projected)
            team_actual[team_display].append(sim_actual)
    
    # Calculate averages
    teams = sorted(team_past.keys(), key=lambda t: np.mean(team_actual[t]), reverse=True)
    
    past_avg = [np.mean(team_past[t]) for t in teams]
    proj_avg = [np.mean(team_projected[t]) for t in teams]
    actual_avg = [np.mean(team_actual[t]) for t in teams]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(18, 10), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')
    
    x = np.arange(len(teams))
    width = 0.25
    
    bars1 = ax.bar(x - width, past_avg, width, label='2023 Performance', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x, proj_avg, width, label='2024 Projected', color='#f39c12', edgecolor='white')
    bars3 = ax.bar(x + width, actual_avg, width, label='2024 Actual', color='#2ecc71', edgecolor='white')
    
    # Add value labels on bars - BIGGER FONT
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, rotation=90, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Formatting - BIGGER FONTS
    ax.set_xlabel("Team (Model)", fontsize=18, fontweight='bold', labelpad=55)
    ax.set_ylabel("Total Fantasy Points from Drafted Players", fontsize=18, fontweight='bold')
    ax.set_title("Draft Strategy Evaluation: Past Performance vs Projections vs Reality\n(Averaged Across 5 Simulations)", 
                fontsize=24, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(" ", "\n") for t in teams], fontsize=13)
    ax.tick_params(axis='y', labelsize=14)
    
    # Set y limits - bars start at 0, no negative space needed
    y_max = max(max(past_avg), max(proj_avg), max(actual_avg)) * 1.15
    ax.set_ylim(0, y_max)
    
    # Add model icons BELOW the plot - use blended transform (data x, axes y)
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, team in enumerate(teams):
        icon = get_model_icon(team, size=70)
        if icon is not None:
            icon_array = np.array(icon)
            zoom_mult = get_icon_zoom_multiplier(team)
            imagebox = OffsetImage(icon_array, zoom=0.55 * zoom_mult)
            # x position is data coordinate (matches tick), y is axes fraction (below plot) - moved down slightly
            ab = AnnotationBbox(imagebox, (i, -0.14), frameon=False,
                               xycoords=trans, box_alignment=(0.5, 0.5),
                               annotation_clip=False)
            ax.add_artist(ab)
    
    legend = ax.legend(loc='upper right', fontsize=14, framealpha=0.95)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "draft_performance.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"  Saved: {output_path}")

# =============================================================================
# Figure 4: Start/Sit Points Left on Table
# =============================================================================

def generate_startsit_analysis():
    """Generate bar chart showing points left on the table from poor start/sit decisions."""
    print("Generating Figure 4: Start/Sit Points Left on Table...")
    
    team_points_lost = defaultdict(list)  # team -> [points lost per sim]
    team_models = {}
    
    for sim_id in SIMULATION_IDS:
        sim_dir = SIMULATIONS_DIR / sim_id
        if not sim_dir.exists():
            continue
        
        weekly_results = load_all_weekly_results(sim_dir)
        rosters = load_draft_rosters(sim_dir)
        team_model_map = get_team_model_mapping(sim_dir)
        
        # Get full roster for each team
        team_full_roster = {}
        for team_name, roster_df in rosters.items():
            team_full_roster[team_name] = set(roster_df["Name"].tolist())
        
        # Calculate points left on table per simulation
        sim_points_lost = defaultdict(float)
        
        for week, df in weekly_results.items():
            # Group by team
            teams_in_week = df["Team"].unique()
            
            for team in teams_in_week:
                if team == "TOTAL" or not team:
                    continue
                
                team_display = normalize_team_name(team)
                
                if team_display not in team_models:
                    team_models[team_display] = team_model_map.get(team, "")
                
                # Get started players this week
                team_rows = df[(df["Team"] == team) & (df["Player"] != "TOTAL") & (df["Position"] != "")]
                started_players = set(team_rows["Player"].tolist())
                started_by_position = defaultdict(list)
                
                for _, row in team_rows.iterrows():
                    pos = row["Position"]
                    pts = float(row["Points"])
                    started_by_position[pos].append((row["Player"], pts))
                
                # Find benched players (in roster but not started)
                full_roster = team_full_roster.get(team, set())
                benched = full_roster - started_players
                
                # For each benched player, check if they outscored a starter at same position
                # We need to estimate bench player points - use their season average / 17
                roster_df = rosters.get(team, pd.DataFrame())
                if roster_df.empty:
                    continue
                
                for _, player_row in roster_df.iterrows():
                    player_name = player_row["Name"]
                    if player_name not in benched:
                        continue
                    
                    pos = player_row["Position"]
                    season_pts = float(player_row.get("FantasyPoints", 0) or 0)
                    weekly_avg = season_pts / 17  # Estimate weekly average
                    
                    # Check if this benched player would have outscored lowest starter at position
                    if pos in started_by_position:
                        starters_at_pos = started_by_position[pos]
                        min_starter = min(starters_at_pos, key=lambda x: x[1])
                        
                        if weekly_avg > min_starter[1]:
                            # Points left on table
                            sim_points_lost[team_display] += (weekly_avg - min_starter[1])
        
        # Store this simulation's results
        for team, pts_lost in sim_points_lost.items():
            team_points_lost[team].append(pts_lost)
    
    # Calculate averages
    teams = sorted(team_points_lost.keys(), key=lambda t: np.mean(team_points_lost[t]))
    avg_lost = [np.mean(team_points_lost[t]) for t in teams]
    
    # Create bar chart - BIGGER SIZE
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')
    
    x = np.arange(len(teams))
    colors = [get_model_color(team_models.get(t, "")) for t in teams]
    
    bars = ax.barh(x, avg_lost, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    
    # Add value labels - BIGGER FONT
    for i, (bar, val) in enumerate(zip(bars, avg_lost)):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
               va='center', fontsize=14, fontweight='bold')
    
    # Formatting - BIGGER FONTS
    ax.set_xlabel("Average Points Left on Table Per Season", fontsize=18, fontweight='bold')
    ax.set_ylabel("Team (Model)", fontsize=18, fontweight='bold', labelpad=55)
    ax.set_title("Start/Sit Decision Quality: Points Left on Bench\n(Averaged Across 5 Simulations — Lower is Better)", 
                fontsize=22, fontweight='bold', pad=20)
    
    ax.set_yticks(x)
    ax.set_yticklabels([t.replace(" ", "\n") for t in teams], fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    
    # Set x limits - bars start at 0, no negative space needed
    ax.set_xlim(0, max(avg_lost) * 1.15)
    
    # Add model icons TO THE LEFT of the plot - use blended transform (axes x, data y)
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i, team in enumerate(teams):
        icon = get_model_icon(team, size=60)
        if icon is not None:
            icon_array = np.array(icon)
            zoom_mult = get_icon_zoom_multiplier(team)
            imagebox = OffsetImage(icon_array, zoom=0.55 * zoom_mult)
            # y position is data coordinate (matches tick), x is axes fraction (left of plot) - moved left slightly
            ab = AnnotationBbox(imagebox, (-0.10, i), frameon=False,
                               xycoords=trans, box_alignment=(0.5, 0.5),
                               annotation_clip=False)
            ax.add_artist(ab)
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add "Better →" annotation
    ax.annotate('← Better', xy=(0.02, 0.98), xycoords='axes fraction',
               fontsize=14, color='#2ecc71', fontweight='bold', va='top')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "startsit_points_lost.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"  Saved: {output_path}")

# =============================================================================
# Figure 5: Trade Value Analysis
# =============================================================================

def generate_trade_value_analysis():
    """Generate bar chart showing average trade value per team."""
    print("Generating Figure 5: Trade Value Analysis...")
    
    team_trade_values = defaultdict(list)  # team -> [trade values]
    team_trade_counts = defaultdict(int)   # team -> total trade count
    team_models = {}
    
    for sim_id in SIMULATION_IDS:
        sim_dir = SIMULATIONS_DIR / sim_id
        if not sim_dir.exists():
            continue
        
        trades = load_accepted_trades(sim_dir)
        weekly_results = load_all_weekly_results(sim_dir)
        team_model_map = get_team_model_mapping(sim_dir)
        
        # Build player -> weekly points map
        player_weekly_points = defaultdict(lambda: defaultdict(float))  # player -> week -> points
        for week, df in weekly_results.items():
            player_rows = df[(df["Player"] != "TOTAL") & (df["Position"] != "")]
            for _, row in player_rows.iterrows():
                player_weekly_points[row["Player"]][week] = float(row["Points"])
        
        # Evaluate each trade
        for trade in trades:
            week = trade.get("week", 1)
            proposer = trade.get("proposer", "")
            receiver = trade.get("receiver", "")
            give = trade.get("give", [])
            receive = trade.get("receive", [])
            
            proposer_display = normalize_team_name(proposer)
            receiver_display = normalize_team_name(receiver)
            
            # Update model mapping
            if proposer_display not in team_models:
                team_models[proposer_display] = team_model_map.get(proposer, "")
            if receiver_display not in team_models:
                team_models[receiver_display] = team_model_map.get(receiver, "")
            
            # Calculate rest-of-season points
            remaining_weeks = range(week + 1, 18)
            
            # Points from players received
            received_pts = sum(
                sum(player_weekly_points[p][w] for w in remaining_weeks)
                for p in receive
            )
            
            # Points from players given away
            given_pts = sum(
                sum(player_weekly_points[p][w] for w in remaining_weeks)
                for p in give
            )
            
            # Trade value = what you got - what you gave
            proposer_value = received_pts - given_pts
            receiver_value = given_pts - received_pts  # Opposite for receiver
            
            team_trade_values[proposer_display].append(proposer_value)
            team_trade_values[receiver_display].append(receiver_value)
            team_trade_counts[proposer_display] += 1
            team_trade_counts[receiver_display] += 1
    
    # Calculate average trade value per team
    teams = sorted(team_trade_values.keys(), 
                   key=lambda t: np.mean(team_trade_values[t]) if team_trade_values[t] else 0)
    
    avg_values = [np.mean(team_trade_values[t]) if team_trade_values[t] else 0 for t in teams]
    trade_counts = [team_trade_counts[t] for t in teams]
    
    # Create bar chart - BIGGER SIZE
    fig, ax = plt.subplots(figsize=(16, 11), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')
    
    x = np.arange(len(teams))
    
    # Color based on value (green for positive, red for negative)
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in avg_values]
    
    bars = ax.barh(x, avg_values, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    
    # Calculate max absolute value for consistent spacing
    max_abs_val = max(abs(v) for v in avg_values) if avg_values else 100
    
    # Add value labels and trade counts - FIXED POSITIONING (moved trade count up)
    for i, (bar, val, count) in enumerate(zip(bars, avg_values, trade_counts)):
        # Value label - position based on sign
        if val >= 0:
            ax.text(val + max_abs_val * 0.03, bar.get_y() + bar.get_height()/2 + 0.08, f'{val:+.1f}',
                   va='center', ha='left', fontsize=14, fontweight='bold')
            # Trade count - just slightly below the value label (moved up)
            ax.text(val + max_abs_val * 0.03, bar.get_y() + bar.get_height()/2 - 0.15, 
                   f'({count} trades)', va='top', ha='left', fontsize=11, color='#666666')
        else:
            ax.text(val - max_abs_val * 0.03, bar.get_y() + bar.get_height()/2 + 0.08, f'{val:+.1f}',
                   va='center', ha='right', fontsize=14, fontweight='bold')
            # Trade count - just slightly below the value label (moved up)
            ax.text(val - max_abs_val * 0.03, bar.get_y() + bar.get_height()/2 - 0.15, 
                   f'({count} trades)', va='top', ha='right', fontsize=11, color='#666666')
    
    # Add vertical line at 0
    ax.axvline(x=0, color='#333333', linewidth=2, linestyle='-')
    
    # Add model icons to the left of y-axis labels - moved to the left
    for i, team in enumerate(teams):
        # Use team name for icon matching
        icon = get_model_icon(team, size=60)
        if icon is not None:
            icon_array = np.array(icon)
            # Apply zoom multiplier
            zoom_mult = get_icon_zoom_multiplier(team)
            imagebox = OffsetImage(icon_array, zoom=0.65 * zoom_mult)
            # Place icon to the left - moved left from 1.25 to 1.5
            ab = AnnotationBbox(imagebox, (-max_abs_val * 1.5, i), frameon=False, 
                               xycoords=('data', 'data'), box_alignment=(1.0, 0.5),
                               annotation_clip=False)
            ax.add_artist(ab)
    
    # Formatting - BIGGER FONTS
    ax.set_xlabel("Average Points Gained Per Trade (Rest-of-Season)", fontsize=18, fontweight='bold')
    ax.set_ylabel("Team (Model)", fontsize=18, fontweight='bold', labelpad=60)
    ax.set_title("Trade Negotiation Effectiveness: Average Value Per Trade\n(Across All 5 Simulations)", 
                fontsize=22, fontweight='bold', pad=20)
    
    ax.set_yticks(x)
    ax.set_yticklabels([t.replace(" ", "\n") for t in teams], fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    
    # Set x limits with more padding on left for icons and right for labels
    ax.set_xlim(-max_abs_val * 1.7, max_abs_val * 1.4)
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Annotations - BIGGER FONT
    ax.annotate('Won Trades →', xy=(0.92, 0.02), xycoords='axes fraction',
               fontsize=14, color='#2ecc71', fontweight='bold', ha='right')
    ax.annotate('← Lost Trades', xy=(0.08, 0.02), xycoords='axes fraction',
               fontsize=14, color='#e74c3c', fontweight='bold', ha='left')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "trade_value.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print(f"  Saved: {output_path}")

# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 60)
    print("FantasyFootballBench Blog Figure Generator")
    print("=" * 60)
    print(f"\nSimulations to analyze: {len(SIMULATION_IDS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Check dependencies
    try:
        import imageio
    except ImportError:
        print("Installing imageio for video generation...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio", "imageio-ffmpeg"])
        import imageio
    
    # Generate all figures
    generate_weekly_standings_video()
    generate_position_distribution()
    generate_draft_performance()
    generate_startsit_analysis()
    generate_trade_value_analysis()
    
    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Check the {OUTPUT_DIR} directory for outputs.")
    print("=" * 60)

if __name__ == "__main__":
    main()

