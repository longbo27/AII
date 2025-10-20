#!/usr/bin/env python3
"""Fetch daily OHLCV data using yfinance based on the configured universe."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml
from loguru import logger
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data" / "raw"
DEFAULT_START = datetime(2010, 1, 1)


def load_universe(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("Universe configuration must be a mapping")
    return data


def download_symbol(symbol: str, start: datetime, end: datetime | None) -> None:
    logger.info("Downloading {} from yfinance", symbol)
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        logger.warning("No data returned for symbol {}", symbol)
        return
    df.index.name = "Date"
    output_path = DATA_DIR / f"{symbol}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logger.info("Saved {} rows to {}", len(df), output_path)


def main(symbols: Iterable[str], start: datetime, end: datetime | None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        try:
            download_symbol(symbol, start, end)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to download {}: {}", symbol, exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OHLCV data for configured universe")
    parser.add_argument("--start", type=str, default=DEFAULT_START.strftime("%Y-%m-%d"))
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None

    universe_cfg = load_universe(CONFIG_DIR / "universe.yml")
    symbols = universe_cfg.get("symbols", [])
    cash_symbol = universe_cfg.get("cash_symbol")
    if cash_symbol:
        symbols = list(dict.fromkeys(symbols + [cash_symbol]))

    logger.info("Fetching data for symbols: {}", symbols)
    main(symbols, start_dt, end_dt)
