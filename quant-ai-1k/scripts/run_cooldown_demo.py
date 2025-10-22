#!/usr/bin/env python3
"""Execute a cooldown stress test and summarize cooldown spans."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"
REPORT_DIR = BASE_DIR / "reports"

RISK_CONFIG_PATH = CONFIG_DIR / "risk.yml"
BACKTEST_COMMAND = [sys.executable, "scripts/run_backtest.py"]
COOLDOWN_SUMMARY_PATH = REPORT_DIR / "cooldown_demo_summary.json"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def run_backtest() -> None:
    result = subprocess.run(
        BACKTEST_COMMAND,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Backtest execution failed with stdout/stderr:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def derive_cooldown_flags(timeseries: pd.DataFrame) -> pd.Series:
    if "cooldown_on" in timeseries.columns:
        return timeseries["cooldown_on"].astype(bool)

    # Heuristic fallback: treat a day as cooldown when exposure is zero
    # immediately after a loss day. This is a best-effort approximation
    # for older report formats that lack an explicit cooldown flag.
    exposure = timeseries.get("exposure", pd.Series(0, index=timeseries.index)).fillna(0)
    net_return = timeseries.get("net_return", pd.Series(0, index=timeseries.index)).fillna(0)
    loss_previous_day = net_return.shift(1) < 0
    zero_exposure = exposure.abs() < 1e-9
    return (loss_previous_day & zero_exposure).astype(bool)


def summarize_windows(cooldown_flags: pd.Series) -> list[dict[str, str]]:
    cooldown_flags = cooldown_flags[cooldown_flags]
    if cooldown_flags.empty:
        return []

    groups = (cooldown_flags.index.to_series().diff().dt.days.ne(1)).cumsum()
    example_windows: list[dict[str, str]] = []
    for _, idx in cooldown_flags.groupby(groups):
        start = idx.index[0]
        end = idx.index[-1]
        example_windows.append(
            {
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
            }
        )
        if len(example_windows) >= 5:
            break
    return example_windows


def main() -> None:
    original_risk = RISK_CONFIG_PATH.read_text(encoding="utf-8")

    try:
        risk_config = load_yaml(RISK_CONFIG_PATH)
        risk_config["daily_loss_limit"] = -0.001
        risk_config["cooldown_days_after_hit"] = 3
        # Maintain legacy hooks used by the D0 engine implementation.
        risk_config["daily_loss_cooldown"] = 0.001
        risk_config["cooldown_days"] = 3
        dump_yaml(RISK_CONFIG_PATH, risk_config)

        run_backtest()

        timeseries_path = REPORT_DIR / "backtest_timeseries.csv"
        if not timeseries_path.exists():
            raise FileNotFoundError(
                "Expected reports/backtest_timeseries.csv after cooldown demo run"
            )

        timeseries = pd.read_csv(timeseries_path, index_col=0, parse_dates=True)
        cooldown_flags = derive_cooldown_flags(timeseries)
        cooldown_flags.index = pd.to_datetime(cooldown_flags.index)
        total_cooldown_days = int(cooldown_flags.sum())
        example_windows = summarize_windows(cooldown_flags)

        summary = {
            "total_cooldown_days": total_cooldown_days,
            "example_windows": example_windows,
        }
        COOLDOWN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        COOLDOWN_SUMMARY_PATH.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        RISK_CONFIG_PATH.write_text(original_risk, encoding="utf-8")


if __name__ == "__main__":
    main()
