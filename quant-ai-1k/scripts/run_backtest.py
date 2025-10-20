#!/usr/bin/env python3
"""Run the backtest end-to-end and persist reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"

sys.path.append(str(BASE_DIR))

from src.backtest.engine import BacktestEngine, load_backtest_config  # noqa: E402
from src.utils.risk import load_risk_config  # noqa: E402


def load_price_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    price_data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = DATA_DIR / f"{symbol}.csv"
        if not path.exists():
            logger.warning("Data file missing for symbol {} at {}", symbol, path)
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        if df.empty:
            logger.warning("Data frame empty for symbol {}", symbol)
            continue
        df = df.set_index("Date").sort_index()
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        if not df.empty:
            start_date = df.index.min()
            end_date = df.index.max()
            logger.info(
                "Loaded data for {} from {} to {}",
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
        price_data[symbol] = df
    if not price_data:
        raise FileNotFoundError("No price data available. Please run scripts/fetch_data.py first.")
    return price_data


def save_reports(timeseries: pd.DataFrame, summary: dict[str, object]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    timeseries_path = REPORT_DIR / "backtest_timeseries.csv"
    timeseries.to_csv(timeseries_path, index=True)
    logger.info("Saved backtest timeseries to {}", timeseries_path)

    summary_path = REPORT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved summary metrics to {}", summary_path)

    equity_path = REPORT_DIR / "equity_curve.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    timeseries["equity_curve"].plot(ax=ax, title="Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (normalized)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(equity_path, dpi=150)
    plt.close(fig)
    logger.info("Saved equity curve plot to {}", equity_path)


def main() -> None:
    risk_config = load_risk_config(CONFIG_DIR / "risk.yml")
    backtest_config, universe_config = load_backtest_config(
        CONFIG_DIR / "backtest.yml", CONFIG_DIR / "universe.yml"
    )

    price_data = load_price_data(universe_config.symbols)
    engine = BacktestEngine(price_data, backtest_config, risk_config)
    results = engine.run()

    save_reports(results["timeseries"], results["summary"])

    logger.info("Summary metrics: {}", json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
