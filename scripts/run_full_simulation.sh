#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   OPENROUTER_API_KEY=... bash scripts/run_full_simulation.sh
# Optional env:
#   FFBENCH_SITE_URL, FFBENCH_SITE_TITLE
# Notes:
#   - Creates a new simulation_{id} and writes all outputs under data/simulations/${FFBENCH_SIM_ID}/

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

# Ensure API key
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set. Export it and re-run." >&2
  exit 1
fi

# Optional metadata for OpenRouter rankings headers
export FFBENCH_SITE_URL="${FFBENCH_SITE_URL:-https://fantasyfootballbench.local}"
export FFBENCH_SITE_TITLE="${FFBENCH_SITE_TITLE:-FantasyFootballBench}"

# Create a fresh simulation id
export FFBENCH_SIM_ID="simulation_$(date +%Y%m%d_%H%M%S)"
echo "Simulation ID: ${FFBENCH_SIM_ID}"

# Write latest_simulation_id.txt early for downstream discovery
SIM_META_DIR="${ROOT}/data/simulations"
mkdir -p "${SIM_META_DIR}"
echo -n "${FFBENCH_SIM_ID}" > "${SIM_META_DIR}/latest_simulation_id.txt"

# Activate venv if present
if [[ -f "${ROOT}/.venv312/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT}/.venv312/bin/activate"
fi

echo "[1/4] Testing model connectivity from config.json..."
python "${ROOT}/scripts/test_models_from_config.py" || true

echo "[2/4] Exporting 2024 projections-based player rankings..."
python "${ROOT}/scripts/export_top_players.py"

echo "[3/4] Running draft simulation..."
python "${ROOT}/scripts/run_draft.py"

echo "[4/4] Simulating full 2024 season (trades + start/sit + matchups)..."
python "${ROOT}/scripts/simulate_season.py"

echo ""
echo "Done. Outputs saved under:"
echo "  ${ROOT}/data/simulations/${FFBENCH_SIM_ID}/draft_results/"
echo "  ${ROOT}/data/simulations/${FFBENCH_SIM_ID}/season_results/"
echo "Simulation ID: ${FFBENCH_SIM_ID}"


