#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

log() {
  printf '[verify_d0] %s\n' "$1"
}

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating Python virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR" || { log "Failed to create virtual environment"; exit 1; }
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate" || { log "Failed to activate virtual environment"; exit 1; }

log "Installing dependencies"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

log "Fetching market data"
python "$ROOT_DIR/scripts/fetch_data.py"

log "Running backtest"
python "$ROOT_DIR/scripts/run_backtest.py"

REPORT_IMG="$ROOT_DIR/reports/equity_curve.png"
if [[ ! -f "$REPORT_IMG" ]]; then
  log "Expected report $REPORT_IMG not found"
  exit 1
fi

if command -v open >/dev/null 2>&1; then
  log "Opening equity curve via macOS open"
  open "$REPORT_IMG"
elif command -v xdg-open >/dev/null 2>&1; then
  log "Opening equity curve via xdg-open"
  xdg-open "$REPORT_IMG"
else
  log "Generated $REPORT_IMG"
fi

log "Verification complete"
