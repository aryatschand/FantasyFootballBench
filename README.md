# Fantasy Football Bench - Complete Season Simulation

A comprehensive fantasy football simulation system that includes drafting, trading, start/sit decisions, and season play using AWS Bedrock LLMs.

## Overview

This system simulates a complete fantasy football season with:
- **10 diverse AI models** competing in a pre-season draft
- **Strategic trading** between teams each week using 4-call negotiation
- **LLM-powered start/sit decisions** based on comprehensive player stats
- **17-week season simulation** with random matchups
- **Comprehensive logging** of all LLM decisions and trades
- **Dynamic roster management** throughout the season

## Prerequisites

1. **Python Environment**:
   ```bash
   python3 -m venv .venv312
   source .venv312/bin/activate
   pip install -r requirements.txt
   ```

2. **AWS Credentials**:
   ```bash
   pip install awscli
   aws configure
   # Enter your AWS credentials when prompted
   ```

3. **Model Configuration**:
   The system uses a `config.json` file to define all simulation parameters. The current configuration includes 10 diverse large language models:
   - Claude 3.5 Sonnet
   - Meta Llama 3.1 405B
   - Claude Sonnet 4
   - Claude Opus 4.1
   - Meta Llama 3.3 70B
   - Meta Llama 3.1 70B
   - Claude 3 Haiku
   - Claude 3.5 Sonnet B
   - Claude Sonnet 4 B
   - DeepSeek-R1

## Step 1: Generate Sorted Player Projections

First, create the sorted player projections file based on 2024 projected stats:

```bash
source .venv312/bin/activate
python scripts/export_top_players.py
```

**What this does:**
- Loads 2024 projected season stats from CSV files (pre-season projections)
- Calculates fantasy points for all players using configurable scoring system
- Resolves abbreviated names to full names using weekly data
- Saves to `data/draft_results/TopPlayers_2024_Projections_PPR.csv`

**Output:** `data/draft_results/TopPlayers_2024_Projections_PPR.csv`

**Scoring System (configurable in config.json):**
- Passing Yards: 0.04 pts/yard
- Passing TDs: 4 pts
- Interceptions: -2 pts
- Rushing Yards: 0.1 pts/yard
- Rushing TDs: 6 pts
- Receptions: 1 pt
- Receiving Yards: 0.1 pts/yard
- Receiving TDs: 6 pts
- Fumbles Lost: -2 pts

## Step 2: Model Configuration

The system is pre-configured with 10 diverse large language models in `config.json`. No additional setup is required for the models.

**Roster Configuration (defined in config.json):**
- 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 5 BENCH spots
- Teams are automatically assigned to models in round-robin fashion

**Current Models:**
- Claude 3.5 Sonnet
- Meta Llama 3.1 405B
- Claude Sonnet 4
- Claude Opus 4.1
- Meta Llama 3.3 70B
- Meta Llama 3.1 70B
- Claude 3 Haiku
- Claude 3.5 Sonnet B
- Claude Sonnet 4 B
- DeepSeek-R1

**Customization:**
To use different models, edit the `models` array in `config.json` with your preferred Bedrock model ARNs.

## Step 3: Run the Draft

Execute the 10-team snake draft with LLM-powered decisions:

```bash
source .venv312/bin/activate
python scripts/run_draft.py
```

**What this does:**
- Each team drafts 10 players (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 5 BENCH)
- Uses 2-step LLM process: suggest 5 candidates from projected rankings → pick 1
- Each model receives roster format and scoring information in prompts
- Logs all LLM calls to JSONL file with full context
- Saves team rosters to individual CSV files with model names

**Draft Process:**
1. **Candidate Selection**: LLM suggests 5 players it wants to research based on projected stats
2. **Stat Research**: System provides detailed 2022/2023 season stats + 2024 projections for the 5 candidates
3. **Final Pick**: LLM makes strategic pick based on roster needs and scoring system
4. **Roster Validation**: Ensures picks fit roster requirements and positions

**Outputs:**
- Team rosters: `data/draft_results/Claude_3.5_Sonnet_Team_1_2024.csv` (etc.)
- LLM logs: `data/draft_results/llm_calls_YYYYMMDD_HHMMSS.jsonl`

## Step 4: Simulate the Season

Run the complete 17-week season with trading and start/sit decisions:

```bash
source .venv312/bin/activate
python scripts/simulate_season.py
```

**What this does:**
- **Week 1-17**: Each week includes:
  1. **Trading Phase**: Each team proposes 1 trade using 4-call negotiation protocol
  2. **Start/Sit Phase**: Each team sets optimal lineup based on comprehensive stats
  3. **Matchup Phase**: Random pairings, calculate fantasy points using actual weekly data
- **4-Call Trade Negotiation**: Propose → Counter/Accept → React → Final Decision
- **Strategic AI Decisions**: Models analyze 2023 season stats, 2024 YTD performance, and recent weekly trends
- **Dynamic Rosters**: Teams evolve through successful trades throughout the season
- **Comprehensive Logging**: All LLM inputs/outputs saved for analysis

**Trading Process:**
1. **Proposal**: Team identifies weakest position and proposes equal trade (1-for-1, 2-for-2, or 3-for-3)
2. **Counter/Accept**: Receiving team can accept or counter with different players
3. **Reaction**: Proposing team can accept counter or make final counter
4. **Final Decision**: Receiving team makes final accept/reject decision

**Start/Sit Process:**
- Models receive: full roster, 2023 season totals, 2024 YTD stats, last 3 weeks performance
- Must select exactly: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (from RB/WR/TE)
- Strategic decisions based on matchups, injuries, and scoring opportunities

**Outputs:**
- Weekly results: `data/season_results/week_X_results.csv`
- Final standings: `data/season_results/final_standings_2024.csv`
- Trade logs: `data/season_results/llm_calls_trades_YYYYMMDD_HHMMSS.jsonl`
- Accepted trades: `data/season_results/accepted_trades_YYYYMMDD_HHMMSS.jsonl`
- Lineup logs: `data/season_results/llm_calls_season_YYYYMMDD_HHMMSS.jsonl`

## Complete Pipeline

To run the entire fantasy football league simulation end-to-end:

```bash
# 1. Activate virtual environment
source .venv312/bin/activate

# 2. Generate player projections based on pre-season 2024 projections
python scripts/export_top_players.py

# 3. Run the 10-team draft with diverse AI models
python scripts/run_draft.py

# 4. Simulate the complete 17-week season with trading and start/sit decisions
python scripts/simulate_season.py
```

**Expected Runtime:**
- **Export**: ~30 seconds (processes ~750 players)
- **Draft**: ~10-15 minutes (10 teams × ~15 rounds × 2 LLM calls each)
- **Season**: ~30-45 minutes (17 weeks × 10 teams × trading + lineup decisions)

**Total Simulation Time:** Approximately 45-60 minutes

## Key Features

- **Full Name Resolution**: Eliminates abbreviation conflicts
- **4-Call Trade Negotiation**: Propose → Counter/Accept → React → Final Decision
- **Strategic AI Decisions**: Models analyze stats and make calculated decisions
- **Comprehensive Logging**: All LLM inputs/outputs saved for analysis
- **Dynamic Rosters**: Teams evolve through trading throughout the season
- **Realistic Simulation**: Uses actual 2024 NFL player data and scoring

## File Structure

```
config.json                              # Simulation configuration (models, scoring, roster)
data/
├── draft_results/
│   ├── TopPlayers_2024_Projections_PPR.csv  # Pre-season projected player rankings
│   ├── Claude_3.5_Sonnet_Team_1_2024.csv    # Team rosters (named by model)
│   └── llm_calls_*.jsonl                   # Draft LLM logs with full context
├── season_results/
│   ├── week_X_results.csv                  # Weekly matchup results with model names
│   ├── final_standings_2024.csv            # Season final standings
│   ├── llm_calls_trades_*.jsonl            # Trade negotiation logs
│   ├── accepted_trades_*.jsonl             # Successful trades log
│   └── llm_calls_season_*.jsonl            # Start/sit decision logs
└── 2024/                                  # Raw NFL data files
    ├── PlayerSeasonStats.csv              # Actual 2024 season data
    ├── PlayerSeasonProjectionStats.csv    # Pre-season 2024 projections
    └── week_X/                            # Weekly game data
scripts/
├── export_top_players.py                  # Generate projections-based rankings
├── run_draft.py                           # Execute draft simulation
├── simulate_season.py                     # Run season with trading/start-sit
└── test_models_from_config.py             # Test model connectivity
ffbench/
├── config.py                              # Configuration management
├── data_handler.py                        # Data access layer
├── llm.py                                 # AWS Bedrock integration
└── scoring.py                             # Fantasy point calculations
```

## Customization

All simulation parameters are configured through `config.json`:

**Models Configuration:**
```json
{
  "models": [
    {"arn": "arn:aws:bedrock:...", "name": "Model Name"},
    ...
  ]
}
```

**Roster Configuration:**
```json
{
  "roster_slots": [
    {"slot": "QB", "count": 1},
    {"slot": "RB", "count": 2},
    {"slot": "WR", "count": 2},
    {"slot": "TE", "count": 1},
    {"slot": "FLEX", "count": 1},
    {"slot": "BENCH", "count": 5}
  ]
}
```

**Scoring Configuration:**
```json
{
  "scoring": {
    "passing_yards": 0.04,
    "passing_tds": 4,
    "interceptions": -2,
    "rushing_yards": 0.1,
    "rushing_tds": 6,
    "receptions": 1,
    "receiving_yards": 0.1,
    "receiving_tds": 6,
    "fumbles_lost": -2
  }
}
```

**Advanced Customization:**
- **Trade Rules**: Modify validation logic in `scripts/simulate_season.py`
- **LLM Prompts**: Update prompt templates in `scripts/run_draft.py` and `scripts/simulate_season.py`
- **Data Processing**: Extend `ffbench/data_handler.py` for additional data sources

## Troubleshooting

**Setup Issues:**
- Ensure AWS credentials are properly configured with `aws configure`
- Check that all data files exist in `data/2024/` directory
- Verify Python virtual environment is activated: `source .venv312/bin/activate`
- Test model connectivity: `python scripts/test_models_from_config.py`

**Configuration Issues:**
- Validate `config.json` format and required fields
- Ensure model ARNs are accessible in your AWS account/region
- Check that roster slots sum to 10 players per team
- Verify scoring weights are numeric values

**Runtime Issues:**
- Monitor LLM call logs in `data/draft_results/llm_calls_*.jsonl` for parsing errors
- Check `data/season_results/llm_calls_trades_*.jsonl` for trade negotiation issues
- Review `data/season_results/llm_calls_season_*.jsonl` for lineup selection problems
- Look for "ValidationException" or "AccessDeniedException" in AWS Bedrock calls

**Data Issues:**
- Ensure `PlayerSeasonProjectionStats.csv` exists for pre-season projections
- Check that weekly data files exist for season simulation (weeks 1-17)
- Verify player name consistency across all data files
- Confirm position data is available for all players

**Performance Tips:**
- Draft simulation: ~10-15 minutes for 10 teams × 15 rounds
- Season simulation: ~30-45 minutes for 17 weeks × 10 teams
- Total runtime: ~45-60 minutes for complete pipeline
- Monitor AWS Bedrock usage and costs during long simulations