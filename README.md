# FantasyFootballBench: A Verifiable LLM Benchmark for Strategic Decision-Making

## Abstract

We present FantasyFootballBench, a novel benchmark for evaluating Large Language Models (LLMs) on fundamental cognitive skills including reasoning, planning, risk assessment, and negotiation. Unlike traditional benchmarks that rely on static question-answering tasks, FantasyFootballBench simulates a complete fantasy football season—a complex, multi-agent strategic environment that requires models to make interdependent decisions with measurable real-world outcomes.

Our benchmark leverages proprietary play-by-play data from the 2024 NFL season (via partnership with SportsData.io) to create a fully verifiable simulation environment. Each LLM controls a fantasy football team and must navigate three critical decision points throughout the season: (1) **draft strategy**, where models select players under roster constraints to maximize projected season-long performance; (2) **weekly start/sit decisions**, where models must optimize lineup composition based on matchup analysis and recent player performance; and (3) **trade negotiations**, where models engage in multi-round bargaining with other LLM-controlled teams to improve roster composition.

The benchmark's key innovation is its verifiability: because we possess complete game-by-game statistics for all NFL players in 2024, we can simulate the exact outcomes of every decision and measure each model's performance against ground truth. This enables precise evaluation of whether models make optimal choices, successfully negotiate mutually beneficial trades, correctly assess injury risk, and adapt strategies based on evolving game dynamics.

FantasyFootballBench addresses critical gaps in current LLM evaluation by testing skills essential for real-world applications—multi-step planning under uncertainty, strategic reasoning with incomplete information, dynamic risk assessment, and collaborative negotiation with other agents. The fantasy football domain provides a natural testbed for these capabilities while maintaining clear success metrics (wins, points scored, roster optimization) and full auditability of all decisions.

This work contributes: (1) a comprehensive benchmark framework for strategic multi-agent decision-making, (2) detailed methodology for evaluating fundamental cognitive skills in LLMs, and (3) a replicable evaluation pipeline with proprietary data access for verified outcomes. Our approach demonstrates how domain-specific benchmarks with complete ground truth can provide deeper insights into LLM capabilities than traditional static evaluation methods.

---

## 1. Introduction and Background

### 1.1 Motivation

Current LLM benchmarks predominantly focus on knowledge retrieval, language understanding, and reasoning over static datasets (e.g., MMLU, HumanEval, GSM8K). While valuable, these benchmarks have significant limitations:

1. **Lack of Strategic Complexity**: Most tasks are single-turn decisions without considering long-term consequences or multi-step planning.
2. **Limited Multi-Agent Interaction**: Few benchmarks evaluate how models negotiate, compete, or cooperate with other agents.
3. **Insufficient Risk Assessment**: Real-world applications require models to evaluate uncertainty, weigh trade-offs, and adapt to changing conditions.
4. **Unverifiable Outcomes**: Many benchmarks rely on subjective human evaluation or lack ground truth for optimal decisions.

FantasyFootballBench addresses these gaps by situating LLM evaluation in the domain of fantasy football—a strategic game that requires:
- **Long-term planning**: Draft decisions compound over a 17-week season.
- **Dynamic adaptation**: Weekly start/sit choices must respond to injuries, matchups, and player trends.
- **Multi-agent negotiation**: Trade proposals require understanding value, persuasion, and counteroffers.
- **Risk management**: Every decision involves uncertainty (player performance, injury probability, opponent strength).

### 1.2 Why Fantasy Football?

Fantasy football provides an ideal benchmark domain for several reasons:

1. **Measurable Outcomes**: Every decision can be evaluated against actual NFL player performance data.
2. **Temporal Dynamics**: The 17-week season creates a sequential decision-making environment where early choices impact later opportunities.
3. **Multi-Agent Competition**: 10 teams compete simultaneously, requiring models to reason about opponent strategies and relative advantage.
4. **Rich Context**: Models must synthesize historical player statistics, injury reports, positional scarcity, and team composition.
5. **Real-World Relevance**: Fantasy sports represent a $9+ billion industry with millions of participants, making the domain culturally significant and well-understood.

### 1.3 Data Partnership and Verifiability

Our benchmark is powered by a proprietary partnership with **SportsData.io**, providing:
- Complete play-by-play statistics for all NFL players (2022–2024 seasons)
- Week-by-week game performance data (passing yards, touchdowns, receptions, etc.)
- Pre-season projection data for the 2024 season
- Historical performance trends for multi-year player analysis

This comprehensive dataset enables **full verifiability**: every simulated outcome in our benchmark corresponds to actual NFL game results. Models are provided only with historical data and projections (no future information), but their decisions are evaluated against ground truth performance. This creates a realistic "information asymmetry" where models must reason under uncertainty, similar to real fantasy football managers.

### 1.4 Core Skills Evaluated

FantasyFootballBench evaluates four fundamental cognitive capabilities:

| Skill | Definition | Evaluation Context |
|-------|-----------|-------------------|
| **Reasoning** | Ability to synthesize multiple data sources, identify patterns, and draw logical conclusions | Draft selection (comparing player projections, injury history, positional value), start/sit optimization (analyzing matchup strength, recent trends) |
| **Planning** | Multi-step strategic thinking with consideration of long-term consequences | Draft strategy (roster construction under constraints), season-long roster management (balancing immediate vs. future value) |
| **Risk Assessment** | Quantifying uncertainty, evaluating trade-offs, and making decisions under incomplete information | Injury evaluation (starting potentially limited players), trade valuation (assessing fair value with performance volatility), bye week planning |
| **Negotiation** | Multi-turn dialogue with adversarial agents to reach mutually beneficial agreements | Trade proposals (crafting compelling offers), counteroffers (evaluating opponent needs), acceptance decisions (determining fair exchanges) |

### 1.5 Benchmark Contributions

This work makes three primary contributions to LLM evaluation:

1. **Framework**: A comprehensive methodology for evaluating strategic decision-making in multi-agent environments with temporal dynamics.
2. **Verifiability**: Complete ground truth for all decisions via proprietary NFL data, enabling objective performance measurement.
3. **Reproducibility**: Open-source simulation pipeline (with data access via SportsData.io partnership) allowing replication across different model sets and configurations.

By focusing on fundamental skills (reasoning, planning, risk, negotiation) rather than domain-specific knowledge, FantasyFootballBench provides insights applicable to broader AI applications including business strategy, resource allocation, and human-AI collaboration systems.

---

## 2. Benchmark Methodology

This section provides a comprehensive, step-by-step walkthrough of the FantasyFootballBench simulation pipeline, detailing every LLM interaction point, the context provided, decisions required, and evaluation metrics.

### 2.1 Simulation Overview

A complete FantasyFootballBench simulation consists of four sequential phases:

1. **Configuration**: Model selection, league setup, and data preparation
2. **Draft Phase**: 12-round snake draft where each model constructs a roster
3. **Season Simulation**: 17 weeks of matchups with weekly start/sit decisions
4. **Trade Negotiation**: Multi-round trade proposals on designated weeks (1, 3, 5, 7, 9)

Each simulation is fully logged with all LLM inputs/outputs saved to JSONL files for reproducibility and detailed analysis.

---

### 2.2 Phase 1: Configuration and Setup

**Purpose**: Establish league structure, model roster, and data environment.

#### 2.2.1 League Configuration

The benchmark uses a standardized 10-team league format configured in `config.json`:

- **10 Teams**: Each controlled by a different LLM (configurable in `models` array)
- **Roster Composition**: 
  - 1 Quarterback (QB)
  - 2 Running Backs (RB)  
  - 2 Wide Receivers (WR)
  - 1 Tight End (TE)
  - 1 FLEX (RB/WR/TE)
  - 5 Bench slots
  - **Total: 12 players per team**
- **Scoring System**: Point-Per-Reception (PPR) format
  - Passing: 0.04 pts/yard, 4 pts/TD, -2 pts/INT
  - Rushing: 0.1 pts/yard, 6 pts/TD
  - Receiving: 1 pt/reception, 0.1 pts/yard, 6 pts/TD
  - Fumbles: -2 pts/fumble lost

#### 2.2.2 Player Pool Generation

Before the draft, the simulation constructs a ranked player pool using 2024 season projection data:

1. **Data Source**: `PlayerSeasonProjectionStats.csv` for 2024
2. **Calculation**: Each player's projected fantasy points computed using the PPR scoring formula
3. **Export**: Top 300+ players by position saved to `TopPlayers_2024_Projections_PPR.csv`
4. **Purpose**: This projection data serves as the "public information" available to all models during draft

**Important**: Models receive projections but NOT actual 2024 outcomes. This creates realistic decision-making under uncertainty.

#### 2.2.3 Model Roster and Draft Order

- **Models Evaluated**: Configurable list of 10 LLMs (e.g., GPT-5, Claude Opus 4.1, Gemini 2.5 Pro, Qwen 3 Max, etc.)
- **Draft Order**: Randomized at simulation start to eliminate position bias
- **Snake Draft Format**: Order reverses each round (e.g., pick 1 in round 1 becomes pick 10 in round 2)

#### 2.2.4 Model Cohort (Evaluated Models)

The current benchmark cohort consists of 10 production-grade models. For clarity and reproducibility, we list both provider/model IDs (used in configs and logs) and human-readable names:

- ID: `openai/gpt-5` — Name: GPT 5
- ID: `openai/gpt-5-mini` — Name: GPT 5 Mini
- ID: `openai/gpt-oss-120b` — Name: GPT OSS
- ID: `anthropic/claude-sonnet-4` — Name: Claude Sonnet 4
- ID: `anthropic/claude-opus-4.1` — Name: Claude Opus 4.1
- ID: `google/gemini-2.5-pro` — Name: Gemini 2.5 Pro
- ID: `google/gemini-2.5-flash` — Name: Gemini 2.5 Flash
- ID: `qwen/qwen3-max` — Name: Qwen 3 Max
- ID: `moonshotai/kimi-k2-0905` — Name: Kimi K2
- ID: `meta-llama/llama-4-maverick` — Name: Llama 4 Maverick

These appear in `config.json` under `models` and are echoed in all draft/season logs.

#### 2.2.5 Runtime and Cost Profile (Per Full Simulation)

Empirically observed totals for a complete run (export → draft → trades + start/sit → season results):

- Total runtime: ~1 hour 45 minutes (±10 minutes depending on provider latency)
- Total LLM cost: ~$7 USD (end-to-end)

Approximate breakdown by phase (may vary with rate limits and retries):

- Draft (12 rounds, 10 teams; ~2 calls/pick ≈ 240 calls): ~$3.50, ~45–55 minutes
- Weekly start/sit (17 weeks × 10 teams ≈ 170 calls): ~$1.50, ~25–30 minutes
- Trades (weeks 1,3,5,7,9; ~10 proposals/week × up to 4 turns ≈ 200 calls, often fewer by early accepts): ~$2.00, ~25–30 minutes

Notes: Costs depend on prompt length and model mix; times depend on concurrency, provider throttling, and occasional retries. The totals above reflect typical runs with the listed cohort.

---

### 2.3 Phase 2: Draft Simulation (12 Rounds)

The draft is the foundation of each team's season and evaluates **reasoning**, **planning**, and **risk assessment** as models construct rosters under constraints.

#### 2.3.1 Draft Structure

- **Format**: Snake draft (12 rounds × 10 teams = 120 total picks)
- **Time**: Each pick involves 2 LLM calls per team (see below)
- **Constraints**: Models must fill all roster slots (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 5 BENCH) by end of draft

#### 2.3.2 LLM Decision Point 1: Candidate Selection

For each pick, the drafting model makes two sequential decisions. First, it selects 5 players to research in detail.

**Context Provided to LLM**:
```
Fantasy Football Draft Setup:
Roster Format: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 5 BENCH (12 total)
Scoring: [PPR format details]

Current roster needs: QB: 0/1, RB: 0/2, WR: 0/2, TE: 0/1, FLEX: 0/1, BENCH: 0/5
Your current roster so far (Name - Position):
- None yet

Below are the suggested top-5 AVAILABLE players by position sorted by projected points 
for the upcoming year (not yet drafted):

Suggested Top QB: Patrick Mahomes, Josh Allen, Lamar Jackson, Dak Prescott, Joe Burrow
Suggested Top RB: Christian McCaffrey, Bijan Robinson, Breece Hall, Jahmyr Gibbs, Saquon Barkley
Suggested Top WR: Justin Jefferson, Tyreek Hill, CeeDee Lamb, Amon-Ra St. Brown, A.J. Brown
Suggested Top TE: Travis Kelce, Sam LaPorta, Mark Andrews, George Kittle, Dalton Kincaid

Return EXACTLY 5 player names you want to research next, comma-separated, 
using the names EXACTLY as shown above.

IMPORTANT: You are ONLY allowed to use the data explicitly provided in this prompt. 
You cannot perform web searches, access external databases, or use any additional 
information not provided here.

Respond ONLY with JSON: {"candidates": ["name1", "name2", "name3", "name4", "name5"]}
```

**Decision Required**: Select 5 players from the provided lists for detailed evaluation.

**Skills Tested**:
- **Reasoning**: Identifying positional need based on current roster state
- **Planning**: Balancing immediate needs vs. best available talent (value-based drafting)
- **Risk Assessment**: Considering positional scarcity (e.g., limited elite TEs)

**Example LLM Response**:
```json
{"candidates": ["Christian McCaffrey", "Bijan Robinson", "Justin Jefferson", "Travis Kelce", "Tyreek Hill"]}
```

#### 2.3.3 LLM Decision Point 2: Final Pick Selection

After selecting 5 candidates, the model receives detailed statistical profiles and must choose exactly one player to draft.

**Context Provided to LLM**:
```
You are drafting.
Your current roster so far (Name - Position):
- None yet

Here are brief summaries for exactly 5 candidates:

- Christian McCaffrey (RB): 2022: 1139 rush yds, 8 rush TDs, 35 rec, 293 rec yds, 3 rec TDs | 
  2023: 1459 rush yds, 14 rush TDs, 67 rec, 564 rec yds, 7 rec TDs | 
  2024 proj: 1640 rush yds, 13 rush TDs, 56 rec, 498 rec yds, 4 rec TDs

- Bijan Robinson (RB): 2022: No data | 
  2023: 976 rush yds, 4 rush TDs, 58 rec, 487 rec yds, 4 rec TDs | 
  2024 proj: 1205 rush yds, 9 rush TDs, 68 rec, 537 rec yds, 3 rec TDs

[... 3 more players with similar detailed context ...]

IMPORTANT: You are ONLY allowed to use the data explicitly provided in this prompt.

Respond ONLY with JSON: {"pick": "<one name from: Christian McCaffrey, Bijan Robinson, Justin Jefferson, Travis Kelce, Tyreek Hill>"}
```

**Decision Required**: Select exactly one player from the 5 candidates to draft.

**Skills Tested**:
- **Reasoning**: Evaluating player consistency across seasons, identifying trends (improvement vs. decline)
- **Planning**: Assessing long-term roster construction (stacking positions vs. balanced approach)
- **Risk Assessment**: Weighing ceiling (2024 projections) vs. floor (2023 actual performance)

**Example LLM Response**:
```json
{"pick": "Christian McCaffrey"}
```

#### 2.3.4 Draft Progression and Context Updates

As the draft progresses, the context provided to each model evolves:

**Round 1 Context** (Example):
```
Current roster needs: QB: 0/1, RB: 0/2, WR: 0/2, TE: 0/1, FLEX: 0/1, BENCH: 0/5
Your current roster so far:
- None yet
```

**Round 6 Context** (Example after 5 picks):
```
Current roster needs: QB: 0/1, RB: 2/2, WR: 1/2, TE: 1/1, FLEX: 0/1, BENCH: 1/5
Your current roster so far (Name - Position):
- Christian McCaffrey (RB)
- Bijan Robinson (RB)
- CeeDee Lamb (WR)
- Travis Kelce (TE)
- Joe Mixon (RB)
```

**Key Mechanism**: The "current roster" is provided to help models understand positional needs and avoid drafting redundant positions. This tests **planning** as models must balance filling mandatory slots vs. stockpiling high-value players.

#### 2.3.5 Draft Constraints and Validation

The simulation enforces roster validity:
- **Positional Limits**: Models cannot draft more players than roster slots allow
- **No Duplicates**: Once a player is drafted by any team, they are removed from the available pool
- **Roster Completion**: By round 12, all teams must have exactly 12 players

If an LLM makes an invalid selection (already drafted, doesn't fit roster), the simulation uses a fallback: select the highest-projected available player that fits the roster.

#### 2.3.6 Draft Outputs

For each pick, the simulation logs:
```json
{
  "time": "2024-10-20T14:23:45",
  "team": "GPT_5_Team_1",
  "call_type": "request_candidates",
  "input": {"prompt": "...full prompt..."},
  "output": {"candidates": ["Player A", "Player B", "Player C", "Player D", "Player E"]}
}
```

```json
{
  "time": "2024-10-20T14:24:12",
  "team": "GPT_5_Team_1",
  "call_type": "final_pick",
  "input": {"prompt": "...full prompt with 5 player summaries..."},
  "output": {"pick": "Player A"}
}
```

All draft logs are saved to `data/simulations/{simulation_id}/draft_results/llm_calls_{timestamp}.jsonl`.

**Final Output**: Each team's complete roster saved to CSV:
```csv
Name,Team,Position,FantasyPoints,Model,ModelName
Christian McCaffrey,SF,RB,285.6,openai/gpt-5,GPT 5
Justin Jefferson,MIN,WR,243.2,openai/gpt-5,GPT 5
...
```

---

### 2.4 Phase 3: Season Simulation (17 Weeks)

After the draft, the simulation advances through 17 weeks of head-to-head matchups. Each week evaluates **reasoning** and **risk assessment** through start/sit decisions.

#### 2.4.1 Weekly Structure

For each of 17 weeks:
1. **Matchup Generation**: Teams randomly paired (5 matchups per week)
2. **Trade Negotiation** (weeks 1, 3, 5, 7, 9 only): See Section 2.5
3. **Lineup Selection**: Each team's LLM chooses starters for the week
4. **Scoring**: Player performances retrieved from actual 2024 NFL data
5. **Results Recording**: Win/loss updated, points accumulated

#### 2.4.2 LLM Decision Point 3: Weekly Lineup Selection

Before each matchup, every team must set a starting lineup: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (from remaining RB/WR/TE).

**Context Provided to LLM** (Example for Week 5):
```
Fantasy Football Start/Sit Decision Setup:
Roster Format: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE)
Scoring: [PPR format details]

Team: GPT_5_Team_1 | Record: 3-1-0

You are setting a fantasy football lineup. 
IMPORTANT: You are ONLY allowed to use the data explicitly provided in this prompt.

Players and stats context:

- Christian McCaffrey (RB): 
  2023 totals: PY=0, PTD=0, INT=0, RY=1459, RTD=14, REC=67, RECY=564, RECTD=7; 
  2024 YTD (weeks 1-4): PY=0, PTD=0, INT=0, RY=459, RTD=4, REC=21, RECY=147, RECTD=1; 
  Last3: wk2: RY=116, RTD=1, REC=6, RECY=39, RECTD=0, wk3: RY=107, RTD=1, REC=4, RECY=20, RECTD=0, wk4: RY=147, RTD=2, REC=5, RECY=38, RECTD=1

- Justin Jefferson (WR): 
  2023 totals: PY=0, PTD=0, INT=0, RY=0, RTD=0, REC=68, RECY=1074, RECTD=5; 
  2024 YTD (weeks 1-4): PY=0, PTD=0, INT=0, RY=0, RTD=0, REC=29, RECY=476, RECTD=3; 
  Last3: wk2: REC=6, RECY=81, RECTD=0, wk3: REC=9, RECY=159, RECTD=2, wk4: REC=8, RECY=118, RECTD=1

- Patrick Mahomes (QB) (INJURED - 0 pts this week): 
  2023 totals: PY=4183, PTD=27, INT=14, RY=389, RTD=2, REC=0, RECY=0, RECTD=0; 
  2024 YTD (weeks 1-4): PY=1015, PTD=8, INT=5, RY=19, RTD=0, REC=0, RECY=0, RECTD=0; 
  Last3: wk2: PY=291, PTD=2, INT=1, wk3: PY=0, PTD=0, INT=0 (DNP), wk4: PY=245, PTD=2, INT=2

[... 9 more players with similar context ...]

Respond ONLY with JSON:
{
  "qb": "Name",
  "rbs": ["Name", "Name"],
  "wrs": ["Name", "Name"],
  "te": "Name",
  "flex": "Name"
}

All names MUST be selected from: Christian McCaffrey, Justin Jefferson, Patrick Mahomes, ...
```

**Key Context Components**:

1. **2023 Season Totals**: Full-season historical performance (baseline capability)
2. **2024 YTD Stats**: Cumulative stats through previous week (current season form)
3. **Last 3 Weeks**: Granular recent performance (hot/cold streaks, consistency)
4. **Injury Flags**: Players scoring 0 points in the current week are explicitly marked as "INJURED - 0 pts this week"

**Decision Required**: Select exactly 1 QB, 2 RBs, 2 WRs, 1 TE, and 1 FLEX from the roster.

**Skills Tested**:
- **Reasoning**: Comparing recent trends vs. historical averages, identifying injury risks
- **Risk Assessment**: Starting players with limited recent sample size vs. proven veterans, evaluating injury probability
- **Planning**: Bench depth management (saving high-value players for future weeks vs. maximizing current week)

**Example LLM Response**:
```json
{
  "qb": "Lamar Jackson",
  "rbs": ["Christian McCaffrey", "Bijan Robinson"],
  "wrs": ["Justin Jefferson", "CeeDee Lamb"],
  "te": "Travis Kelce",
  "flex": "Breece Hall"
}
```

#### 2.4.3 Injury Handling and Zero-Point Players

A critical aspect of start/sit decisions is injury management. The simulation identifies injured players by checking if their week N actual points = 0.0. These players are flagged in the prompt with `(INJURED - 0 pts this week)`.

**Purpose**: Tests whether models can interpret injury signals and adjust lineups accordingly. This evaluates **risk assessment** and **reasoning** under incomplete information (the prompt doesn't explicitly say "do not start injured players," requiring the model to infer this).

#### 2.4.4 Lineup Validation and Fallback

If the LLM returns an invalid lineup (wrong positions, duplicate players, missing slots), the simulation uses a **best-available fallback**:
1. Rank all roster players by projected points for the week
2. Fill each position slot with the highest-ranked available player
3. Use this heuristic lineup for scoring

This ensures the simulation always completes while penalizing models that fail to follow instructions.

#### 2.4.5 Weekly Scoring and Results

Once lineups are set, the simulation:
1. Retrieves actual Week N performance for each started player from `PlayerGameStatsByWeek.csv`
2. Calculates fantasy points using the PPR formula
3. Sums total points for each team
4. Determines winner (higher score) and updates standings

**Example Weekly Results** (`week_5_results.csv`):
```csv
Week,Team,Model,Opponent,Player,ResolvedName,Position,Points
5,GPT_5_Team_1,openai/gpt-5,Claude_Sonnet_4_Team_2,Christian McCaffrey,Christian McCaffrey,RB,18.7
5,GPT_5_Team_1,openai/gpt-5,Claude_Sonnet_4_Team_2,Justin Jefferson,Justin Jefferson,WR,23.4
5,GPT_5_Team_1,openai/gpt-5,Claude_Sonnet_4_Team_2,Lamar Jackson,Lamar Jackson,QB,19.2
...
5,GPT_5_Team_1,openai/gpt-5,Claude_Sonnet_4_Team_2,TOTAL,,,,112.3
5,Claude_Sonnet_4_Team_2,anthropic/claude-sonnet-4,GPT_5_Team_1,TOTAL,,,,98.7
```

#### 2.4.6 Standings Tracking

After each week, the simulation updates a running standings table:
```csv
Team,Wins,Losses,Ties,PointsFor
GPT_5_Team_1,4,1,0,524.3
Claude_Sonnet_4_Team_2,3,2,0,487.9
...
```

**Evaluation Metrics**:
- **Win Rate**: Primary measure of overall performance
- **Points For**: Total fantasy points scored across all weeks (measures lineup optimization quality)
- **Start/Sit Accuracy**: Percent of weeks where started players outscored bench players at the same position (see Section 3)

---

### 2.5 Phase 4: Trade Negotiation (Weeks 1, 3, 5, 7, 9)

Trades are the most complex decision point, evaluating **negotiation**, **reasoning**, and **risk assessment** through multi-turn dialogue between LLM-controlled teams.

#### 2.5.1 Trade Structure

On designated trade weeks (1, 3, 5, 7, 9), each team initiates one trade proposal:
1. **Proposer** identifies a target team and proposes a trade (1-for-1, 2-for-2, or 3-for-3)
2. **Receiver** evaluates the proposal and either **accepts** or **counters**
3. **Proposer** evaluates the counter and either **accepts** or **counters again**
4. **Receiver** makes a final **accept/reject** decision

This 4-stage negotiation simulates realistic trade dynamics with multiple rounds of offers/counteroffers.

#### 2.5.2 LLM Decision Point 4: Trade Proposal Generation

Each team, acting as proposer, must craft a trade offer to another team.

**Context Provided to LLM** (Example for Week 3):
```
Fantasy Football Trade Negotiation Setup:
Roster Format: [roster details]
Scoring: [PPR details]

IMPORTANT: You are ONLY allowed to use the data explicitly provided in this prompt.

You are proposing a fantasy football trade.
Your team: GPT_5_Team_1 (Record: 2-0-0) (weakest position heuristic: TE).

Your roster with stats:
- Christian McCaffrey (RB) | 2024 YTD to wk2: RY=273, RTD=2, REC=15, RECY=126, RECTD=1; Last wk: RY=157, RTD=1, REC=9, RECY=87, RECTD=1
- Justin Jefferson (WR) | 2024 YTD to wk2: RY=0, RTD=0, REC=13, RECY=198, RECTD=1; Last wk: REC=7, RECY=117, RECTD=1
- George Kittle (TE) | 2024 YTD to wk2: RY=0, RTD=0, REC=4, RECY=39, RECTD=0; Last wk: REC=2, RECY=15, RECTD=0
[... full roster ...]

Other teams and their rosters with stats (records shown in headers):

Team Claude_Sonnet_4_Team_2 (Record: 1-1-0) roster:
- Travis Kelce (TE) | 2024 YTD to wk2: RY=0, RTD=0, REC=10, RECY=117, RECTD=2; Last wk: REC=5, RECY=67, RECTD=1
- Davante Adams (WR) | 2024 YTD to wk2: RY=0, RTD=0, REC=11, RECY=142, RECTD=1; Last wk: REC=6, RECY=76, RECTD=1
[... full roster ...]

[... 8 more teams with full rosters and stats ...]

Propose ONE trade with EXACTLY equal number of players on both sides (1-for-1, 2-for-2, or 3-for-3). 
Prefer 2-for-2 if both teams have depth. NEVER propose uneven trades.

Include positions and a convincing rationale for the other team.

Return ONLY JSON:
{
  "target_team": "TeamName",
  "give": [{"name": "Full Name", "position": "POS"}, ...],
  "receive": [{"name": "Full Name", "position": "POS"}, ...],
  "rationale": "why this helps them"
}
```

**Decision Required**: 
1. Select a target team
2. Identify players to trade away from own roster
3. Identify players to receive from target team
4. Craft a persuasive rationale for why the trade benefits the target team

**Skills Tested**:
- **Reasoning**: Identifying own team weaknesses (e.g., "weakest position: TE" hint), evaluating relative player values
- **Planning**: Assessing roster construction needs (depth vs. star power, positional balance)
- **Negotiation**: Framing the trade to emphasize benefits for the opponent (persuasion)
- **Risk Assessment**: Trading away known value for uncertain upside, evaluating injury-prone players

**Example LLM Response**:
```json
{
  "target_team": "Claude_Sonnet_4_Team_2",
  "give": [
    {"name": "George Kittle", "position": "TE"},
    {"name": "Deebo Samuel Sr.", "position": "WR"}
  ],
  "receive": [
    {"name": "Travis Kelce", "position": "TE"},
    {"name": "Davante Adams", "position": "WR"}
  ],
  "rationale": "This trade gives you two WR1-caliber players in Deebo and DK while upgrading your TE position with Kittle's consistent production. Kelce's age and declining targets make this a win-now move for you."
}
```

#### 2.5.3 LLM Decision Point 5: Trade Evaluation (Receiver)

The target team receives the proposal and must decide: **accept** or **counter**.

**Context Provided to LLM**:
```
Fantasy Football Trade Negotiation Setup:
[roster format, scoring details]

IMPORTANT: You are ONLY allowed to use the data explicitly provided in this prompt.

You are the receiver of a trade proposal. 
Your team: Claude_Sonnet_4_Team_2 (Record: 1-1-0).

Proposal: you would RECEIVE ["George Kittle", "Deebo Samuel Sr."] and GIVE ["Travis Kelce", "Davante Adams"]. 
Rationale from proposer: "This trade gives you two WR1-caliber players..."

Proposer roster with stats:
[full proposer roster with YTD and last week stats]

Receiver roster with stats:
[full receiver roster with YTD and last week stats]

If the proposal clearly benefits your team (improves starters, depth, or expected points), choose accept. 
If not, respond with a COUNTER of EXACTLY equal size. NEVER propose uneven trades.

Respond ONLY JSON. Either:
- Accept: {"decision": "accept"}
- Counter: {"decision": "counter", "give": [{"name":"...", "position":"..."}], "receive": [{"name":"...", "position":"..."}], "rationale": "..."}
```

**Decision Required**: Evaluate the proposed trade and either accept or construct a counteroffer of equal size.

**Skills Tested**:
- **Reasoning**: Comparing player projections, evaluating positional scarcity, assessing team needs
- **Negotiation**: Deciding when an offer is "good enough" vs. when to push for better terms
- **Risk Assessment**: Weighing immediate value vs. future uncertainty (e.g., injury risk, playoff schedules)

**Example LLM Response (Counter)**:
```json
{
  "decision": "counter",
  "give": [
    {"name": "Davante Adams", "position": "WR"},
    {"name": "Alvin Kamara", "position": "RB"}
  ],
  "receive": [
    {"name": "Christian McCaffrey", "position": "RB"},
    {"name": "Deebo Samuel Sr.", "position": "WR"}
  ],
  "rationale": "I'm willing to trade Adams but need a true RB1 upgrade. McCaffrey's elite production justifies giving up Kamara and Adams."
}
```

**Example LLM Response (Accept)**:
```json
{
  "decision": "accept"
}
```

#### 2.5.4 LLM Decision Point 6: Counter Evaluation (Proposer)

If the receiver counters, the original proposer must evaluate the counteroffer.

**Context Provided to LLM**:
```
Fantasy Football Trade Negotiation Setup:
[roster format, scoring]

You are the original proposer evaluating a counter. 
Your team: GPT_5_Team_1 (Record: 2-0-0).

Counter proposal: YOU would GIVE ["Christian McCaffrey", "Deebo Samuel Sr."] and RECEIVE ["Davante Adams", "Alvin Kamara"]. 
Rationale from other team: "I'm willing to trade Adams but need a true RB1 upgrade..."

If the counter clearly benefits your team, choose accept. 
If not, respond with ONE final COUNTER of EXACTLY equal size. NEVER propose uneven trades.

Respond ONLY JSON. Either:
- Accept: {"decision": "accept"}
- Counter: {"decision": "counter", "give": [...], "receive": [...], "rationale": "..."}
```

**Decision Required**: Evaluate the counter and either accept or make a final counteroffer.

**Skills Tested**: Same as Decision Point 5, plus:
- **Negotiation**: Knowing when to "walk away" from a deal vs. compromise
- **Planning**: Considering how the trade impacts future lineup flexibility

**Example LLM Response (Counter)**:
```json
{
  "decision": "counter",
  "give": [
    {"name": "Bijan Robinson", "position": "RB"},
    {"name": "Deebo Samuel Sr.", "position": "WR"}
  ],
  "receive": [
    {"name": "Davante Adams", "position": "WR"},
    {"name": "Travis Kelce", "position": "TE"}
  ],
  "rationale": "I'll trade Bijan instead of CMC. Adding Kelce addresses my TE need while upgrading WR depth."
}
```

#### 2.5.5 LLM Decision Point 7: Final Accept/Reject (Receiver)

The receiver makes a final binding decision on the proposer's second counter.

**Context Provided to LLM**:
```
Final decision. Evaluate this final counter. 
Your team: Claude_Sonnet_4_Team_2 (Record: 1-1-0).

Final counter: you would GIVE ["Davante Adams", "Travis Kelce"] and RECEIVE ["Bijan Robinson", "Deebo Samuel Sr."].

If this improves your team, choose accept; otherwise reject. NEVER accept uneven trades.

Respond ONLY JSON. Either:
- {"decision": "accept"}
- {"decision": "reject"}
```

**Decision Required**: Final accept/reject (no further negotiation possible).

**Skills Tested**:
- **Negotiation**: Assessing whether incremental improvements justify agreement
- **Risk Assessment**: Making a binary decision under uncertainty

**Example LLM Response**:
```json
{
  "decision": "accept"
}
```

#### 2.5.6 Trade Execution and Roster Updates

If any stage results in "accept":
1. Players are swapped between team rosters
2. Team CSV files are updated with new rosters
3. Trade is logged to `accepted_trades_{timestamp}.jsonl`:
```json
{
  "time": "2024-10-20T15:34:12",
  "week": 3,
  "accepted_stage": "final_decision",
  "proposer": "GPT_5_Team_1",
  "receiver": "Claude_Sonnet_4_Team_2",
  "give": ["Bijan Robinson", "Deebo Samuel Sr."],
  "receive": ["Davante Adams", "Travis Kelce"]
}
```

If all 4 stages complete without acceptance, the trade is rejected (no roster changes).

#### 2.5.7 Trade Constraints and Validation

The simulation enforces trade validity:
- **Equal Counts**: Must be N-for-N (1-for-1, 2-for-2, or 3-for-3)
- **Ownership**: All "give" players must be on proposer's roster, all "receive" players on receiver's roster
- **Maximum Size**: No more than 3 players per side

Invalid trades are auto-rejected or replaced with a valid fallback trade.

#### 2.5.8 Trade Negotiation Outputs

All trade negotiations are logged to `llm_calls_trades_{timestamp}.jsonl`:
```json
{
  "time": "2024-10-20T15:30:45",
  "stage": "propose",
  "week": 3,
  "proposer": "GPT_5_Team_1",
  "receiver": "Claude_Sonnet_4_Team_2",
  "output": {"target_team": "Claude_Sonnet_4_Team_2", "give": [...], "receive": [...], "rationale": "..."}
}
```

This enables detailed analysis of:
- **Negotiation patterns**: How many rounds until acceptance? Who initiates successful trades?
- **Valuation accuracy**: Do models correctly assess player value?
- **Persuasion effectiveness**: Does rationale quality correlate with acceptance rate?

---

### 2.6 Simulation Outputs and Data Artifacts

Each simulation generates a comprehensive directory structure under `data/simulations/{simulation_id}/`:

#### 2.6.1 Draft Results (`draft_results/`)
- `llm_calls_{timestamp}.jsonl`: All draft LLM interactions (candidate selection + final picks)
- `{Team_Name}_2024.csv`: Final drafted roster for each team
- `TopPlayers_2024_Projections_PPR.csv`: Projection data used during draft

#### 2.6.2 Season Results (`season_results/`)
- `llm_calls_season_{timestamp}.jsonl`: All weekly start/sit LLM interactions
- `llm_calls_trades_{timestamp}.jsonl`: All trade negotiation LLM interactions
- `accepted_trades_{timestamp}.jsonl`: Log of executed trades
- `week_{1-17}_results.csv`: Player-level points and matchup outcomes for each week
- `final_standings_2024.csv`: Final win/loss records and total points

#### 2.6.3 Log Format

All LLM logs follow a consistent JSONL structure:
```json
{
  "time": "ISO 8601 timestamp",
  "team": "Team identifier",
  "call_type": "request_candidates | final_pick | start_sit | trade_propose | ...",
  "input": {"prompt": "Full prompt text sent to LLM"},
  "output": {"Parsed JSON response from LLM"}
}
```

This enables:
- **Reproducibility**: Exact prompts can be replayed or analyzed
- **Debugging**: Failed responses can be inspected
- **Ablation Studies**: Prompt engineering experiments by modifying inputs

---

### 2.7 Evaluation Metrics (Overview)

The benchmark provides multiple evaluation dimensions (detailed in Section 3):

#### 2.7.1 Primary Metrics
- **Win Rate**: Games won / total games played (17 weeks)
- **Points For**: Total fantasy points scored across season
- **Draft Value**: Sum of actual 2024 PPR from drafted players vs. projections

#### 2.7.2 Decision Quality Metrics
- **Start/Sit Accuracy**: % of weeks where optimal lineup was selected
- **Trade Win Rate**: Cumulative value gained/lost from accepted trades
- **Positional Balance**: Variance in positional strength (balanced vs. overweighted)

#### 2.7.3 Behavioral Metrics
- **Draft Strategy**: Early RB priority vs. WR priority, TE draft round
- **Negotiation Style**: Acceptance rate when proposing, countering frequency
- **Risk Tolerance**: Starting high-variance players vs. safe floors

---

## 3. Key Design Principles

### 3.1 Information Asymmetry and Realism

A critical aspect of FantasyFootballBench is that **models receive only historically available information**:
- **Draft Phase**: Models see 2022-2023 actual stats and 2024 projections (no 2024 outcomes)
- **Week N Lineup**: Models see weeks 1 through N-1 actual stats (no week N outcomes)
- **Trade Negotiations**: Models see YTD stats through previous week (no future performance)

This mirrors real-world fantasy football where managers make decisions under uncertainty. The benchmark evaluates **reasoning under incomplete information**, not memorization of outcomes.

### 3.2 Verifiable Ground Truth

Unlike subjective evaluation benchmarks, every decision in FantasyFootballBench has **objective ground truth**:
- **Draft**: Did the player perform better or worse than projection in actual 2024 season?
- **Start/Sit**: Did the started player score more points than bench alternatives?
- **Trades**: Did the acquired players outscore the traded-away players in subsequent weeks?

This enables precise quantification of decision quality without human judgment.

### 3.3 Multi-Agent Competition

Unlike single-agent benchmarks, FantasyFootballBench features **competitive multi-agent dynamics**:
- **Relative Performance**: Winning requires outperforming other LLMs, not just an absolute threshold
- **Strategic Interaction**: Draft picks and trades directly impact opponent resources
- **Emergent Behavior**: Negotiation outcomes depend on both agents' strategies

This tests whether models can reason about adversarial agents and adapt strategies accordingly.

### 3.4 Temporal Dependencies

Decisions compound over time:
- **Draft mistakes** (e.g., injury-prone player) impact all 17 weeks
- **Bad trades** permanently alter roster composition
- **Start/sit errors** accumulate into season-long point deficits

This tests **long-term planning** and ability to recover from early mistakes.

### 3.5 No External Tool Use

All LLM interactions explicitly prohibit external tool use:
> "IMPORTANT: You are ONLY allowed to use the data explicitly provided in this prompt. You cannot perform web searches, access external databases, or use any additional information not provided here."

This ensures the benchmark evaluates **pure reasoning capabilities** using only in-context information, not retrieval augmentation or function calling.

---

## 4. Reproducibility and Extensions

### 4.1 Running the Benchmark

Full simulations are executed via the pipeline script:
```bash
export OPENROUTER_API_KEY="your_api_key"
bash scripts/run_full_simulation.sh
```

This sequentially runs:
1. Model connectivity test (`test_models_from_config.py`)
2. Projection data export (`export_top_players.py`)
3. Draft simulation (`run_draft.py`)
4. Season simulation (`simulate_season.py`)

### 4.2 Configuration

All league parameters are centralized in `config.json`:
- `models`: List of LLM identifiers to evaluate
- `roster_slots`: Positional requirements
- `scoring`: PPR point values
- `trade_weeks`: Specific weeks for trade negotiations
- `season_weeks`: Number of weeks to simulate (17 for full season)

Researchers can easily modify these to test alternative league formats (e.g., standard vs. PPR scoring, different roster sizes).

### 4.3 Data Access

The benchmark requires access to SportsData.io proprietary data (2022-2024 NFL seasons). Researchers interested in replicating the benchmark can:
1. Obtain a SportsData.io API key (partnership or paid access)
2. Use the provided data scraper (`scripts/data_scraper.py`) to populate local data files
3. Run the full simulation pipeline

### 4.4 Extending the Benchmark

Potential extensions include:
- **Auction Draft Format**: Test bidding strategies and budget management
- **Dynasty Leagues**: Multi-season evaluation with rookie drafts and keeper decisions
- **Playoff Simulation**: Weeks 15-17 become single-elimination tournament
- **In-Season Injuries**: Dynamic roster updates requiring waiver wire pickups
- **Opponent-Aware Lineup Selection**: Provide opposing team's lineup before start/sit decision

---

## 5. Limitations and Future Work

### 5.1 Current Limitations

1. **Single Sport**: Fantasy football is US-centric; international applicability unclear
2. **Data Cost**: Proprietary data limits open-source reproducibility
3. **Compute Requirements**: ~500+ LLM calls per simulation (cost and time intensive)
4. **No Mid-Season Adjustments**: Waiver wire pickups not yet implemented

### 5.2 Future Directions

1. **Multi-Sport Expansion**: Fantasy basketball, soccer, baseball benchmarks
2. **Human Baselines**: Compare LLM performance to expert human fantasy managers
3. **Explainability Analysis**: Evaluate whether models provide sound rationales for decisions
4. **Adversarial Evaluation**: Test robustness to misleading statistics or biased projections

---

## 6. Conclusion

FantasyFootballBench provides a rigorous, verifiable benchmark for evaluating fundamental LLM capabilities in strategic decision-making. By situating evaluation in a complex multi-agent environment with temporal dynamics and incomplete information, the benchmark offers insights beyond traditional static evaluations.

The framework's key contributions are:
1. **Verifiability** via complete 2024 NFL ground truth data
2. **Skill decomposition** across reasoning, planning, risk assessment, and negotiation
3. **Realism** through information asymmetry and multi-agent competition
4. **Reproducibility** with open-source simulation pipeline

We hope FantasyFootballBench serves as a model for future benchmarks that evaluate AI systems in realistic, high-stakes decision-making scenarios.

---

## Citation

If you use FantasyFootballBench in your research, please cite:

```bibtex
@misc{fantasyfootballbench2024,
  title={FantasyFootballBench: A Verifiable LLM Benchmark for Strategic Decision-Making},
  author={[Authors]},
  year={2024},
  note={Data partnership with SportsData.io}
}
```

---

## Acknowledgments

This research was made possible through a data partnership with **SportsData.io**, which provided proprietary NFL statistics for the 2022-2024 seasons. We thank the SportsData.io team for their support in enabling verifiable sports analytics research.

---

## License

[To be determined based on data partnership agreements and code licensing preferences]
