#!/usr/bin/env bash
set -euo pipefail

# Run the full FantasyFootballBench pipeline: connectivity check -> projections
# export -> draft -> 17-week season.
#
# Usage:
#   bash scripts/run_full_simulation.sh
#
# Configuration comes from .env (see .env.example) or the surrounding shell.
# Required: OPENROUTER_API_KEY
# Optional: FFBENCH_SITE_URL, FFBENCH_SITE_TITLE, FFBENCH_CONFIG, FFBENCH_SIM_ID
#
# All outputs land in data/simulations/${FFBENCH_SIM_ID}/.

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
cd "${ROOT}"

# Load .env if present (real environment variables take precedence).
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set." >&2
  echo "       Copy .env.example to .env and add your key, or export it." >&2
  exit 1
fi

export FFBENCH_SITE_URL="${FFBENCH_SITE_URL:-https://fantasyfootballbench.local}"
export FFBENCH_SITE_TITLE="${FFBENCH_SITE_TITLE:-FantasyFootballBench}"

# Prefer the project venv if one exists.
for candidate in "${ROOT}/.venv/bin/activate" "${ROOT}/.venv312/bin/activate"; do
  if [[ -f "${candidate}" ]]; then
    # shellcheck disable=SC1090
    source "${candidate}"
    break
  fi
done

exec python main.py all "$@"
