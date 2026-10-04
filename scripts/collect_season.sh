#!/usr/bin/env bash
# One tick of the season collector: run `pramaan collect` only inside the results window and
# only one pass at a time. Called every 30 minutes by the systemd user timer
# (scripts/systemd/pramaan-collect.{service,timer}). Outside the window it does nothing.
#
# The season writes to its own home so the dry run's frozen splits and results stay untouched.
# Override with PRAMAAN_SEASON_HOME, PRAMAAN_SEASON_START, PRAMAAN_SEASON_END (YYYY-MM-DD).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START="${PRAMAAN_SEASON_START:-2026-10-12}"
END="${PRAMAAN_SEASON_END:-2026-11-20}"
today="$(date +%F)"
if [[ "$today" < "$START" || "$today" > "$END" ]]; then
  echo "$(date -Is) outside the season window ($START to $END): nothing to do"
  exit 0
fi

export PRAMAAN_HOME="${PRAMAAN_SEASON_HOME:-$HOME/.local/share/pramaan-season}"
mkdir -p "$PRAMAAN_HOME"
exec 9>"$PRAMAAN_HOME/collect.lock"
if ! flock -n 9; then
  echo "$(date -Is) a previous pass is still running: skipping this tick"
  exit 0
fi

cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
echo "$(date -Is) collecting into $PRAMAAN_HOME"
pramaan collect --profile q2fy27_season --others 150 --reference-pdf
