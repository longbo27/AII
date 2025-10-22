#!/usr/bin/env bash
set -euo pipefail

python scripts/validate_config.py
python scripts/run_regression.py
python scripts/run_cooldown_demo.py

echo "== Done. Check reports/regression/grid_summary.csv and reports/cooldown_demo_summary.json =="
