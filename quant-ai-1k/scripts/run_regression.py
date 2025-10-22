#!/usr/bin/env python3
"""Run the backtest across a grid of rebalance frequencies and cost presets."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"
REPORT_DIR = BASE_DIR / "reports"
REGRESSION_DIR = REPORT_DIR / "regression"
BACKTEST_SCRIPT = [sys.executable, "scripts/run_backtest.py"]

BACKTEST_CONFIG_PATH = CONFIG_DIR / "backtest.yml"
RISK_CONFIG_PATH = CONFIG_DIR / "risk.yml"

REBALANCE_FREQUENCIES = ("daily", "weekly", "monthly")
COST_PRESETS: dict[str, dict[str, float]] = {
    "low": {"slippage_bps": 5.0, "commission_per_share": 0.0025},
    "med": {"slippage_bps": 10.0, "commission_per_share": 0.0050},
    "high": {"slippage_bps": 20.0, "commission_per_share": 0.0100},
}

SUMMARY_COLUMNS = [
    "rebalance_freq",
    "cost_preset",
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "ann_vol",
    "trading_days",
]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    """Persist a YAML mapping to ``path`` preserving key order."""

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def run_backtest() -> None:
    """Execute the canonical backtest script."""

    result = subprocess.run(
        BACKTEST_SCRIPT,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = (
            "Backtest execution failed with stdout/stderr:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        raise RuntimeError(message)


def copy_artifacts(destination: Path) -> None:
    """Copy key backtest artifacts into ``destination`` if they exist."""

    destination.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "summary.json": "summary.json",
        "equity_curve.png": "equity_curve.png",
        "monthly_returns.csv": "monthly_returns.csv",
        "annual_returns.csv": "annual_returns.csv",
    }

    for source_name, target_name in artifacts.items():
        source_path = REPORT_DIR / source_name
        if source_path.exists():
            shutil.copy2(source_path, destination / target_name)


def collect_summary(summary_path: Path) -> dict[str, Any]:
    """Read a summary JSON file and expose required metrics."""

    with summary_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Summary at {summary_path} is not a mapping")
    return data


def main() -> None:
    original_backtest = BACKTEST_CONFIG_PATH.read_text(encoding="utf-8")
    original_risk = RISK_CONFIG_PATH.read_text(encoding="utf-8")

    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
    grid_records: list[dict[str, Any]] = []

    try:
        for freq, preset_name in product(REBALANCE_FREQUENCIES, COST_PRESETS.keys()):
            cost_values = COST_PRESETS[preset_name]
            print(f"[RUN] freq={freq} cost={preset_name}")

            backtest_config = load_yaml(BACKTEST_CONFIG_PATH)
            risk_config = load_yaml(RISK_CONFIG_PATH)

            backtest_config["rebalance_freq"] = freq
            backtest_config["rebalance"] = freq  # D0 compatibility

            risk_config["slippage_bps"] = float(cost_values["slippage_bps"])
            risk_config["commission_per_share"] = float(cost_values["commission_per_share"])

            # Ensure legacy cost hooks stay aligned with the D1 values.
            backtest_config["slippage_bps"] = float(cost_values["slippage_bps"])
            backtest_config["commission_per_trade"] = float(cost_values["commission_per_share"])

            dump_yaml(BACKTEST_CONFIG_PATH, backtest_config)
            dump_yaml(RISK_CONFIG_PATH, risk_config)

            run_backtest()

            destination = REGRESSION_DIR / f"{freq}_{preset_name}"
            if destination.exists():
                shutil.rmtree(destination)
            copy_artifacts(destination)

            summary_path = destination / "summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Expected summary.json at {summary_path}")

            summary = collect_summary(summary_path)
            record = {
                "rebalance_freq": freq,
                "cost_preset": preset_name,
            }
            for key in SUMMARY_COLUMNS[2:]:
                record[key] = summary.get(key)
            grid_records.append(record)

        if grid_records:
            df = pd.DataFrame(grid_records, columns=SUMMARY_COLUMNS)
            grid_summary_path = REGRESSION_DIR / "grid_summary.csv"
            df.to_csv(grid_summary_path, index=False)
        else:
            raise RuntimeError("No regression results collected; check backtest outputs.")
    finally:
        BACKTEST_CONFIG_PATH.write_text(original_backtest, encoding="utf-8")
        RISK_CONFIG_PATH.write_text(original_risk, encoding="utf-8")


if __name__ == "__main__":
    main()
