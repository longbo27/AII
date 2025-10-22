#!/usr/bin/env python3
"""Validate required configuration keys and ranges for the D1 stack."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"

RISK_CONFIG_PATH = CONFIG_DIR / "risk.yml"
BACKTEST_CONFIG_PATH = CONFIG_DIR / "backtest.yml"


class ValidationError(Exception):
    """Raised when a configuration validation fails."""


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValidationError(f"{path} must contain a mapping")
    return data


def require_keys(data: dict[str, Any], keys: list[str], context: str) -> None:
    for key in keys:
        if key not in data:
            raise ValidationError(f"Missing required key '{key}' in {context}")


def get_float(data: dict[str, Any], key: str) -> float:
    try:
        return float(data[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"Key '{key}' in risk.yml must be numeric") from exc


def ensure_range(value: float, lower: float, upper: float, key: str) -> None:
    if not (lower <= value <= upper):
        raise ValidationError(
            f"{key} must be between {lower} and {upper}, got {value}"
        )


def validate_risk_config(risk_config: dict[str, Any]) -> None:
    required_keys = [
        "max_drawdown_hard",
        "max_drawdown_soft",
        "position_max_gross",
        "position_max_per_asset",
        "risk_per_trade",
        "daily_loss_limit",
        "cooldown_days_after_hit",
        "slippage_bps",
        "commission_per_share",
        "stop_mode",
        "stop_atr_mult",
        "takeprofit_rr",
    ]
    require_keys(risk_config, required_keys, "risk.yml")

    max_dd_hard = get_float(risk_config, "max_drawdown_hard")
    ensure_range(max_dd_hard, -1.0, -0.05, "max_drawdown_hard")

    max_dd_soft = get_float(risk_config, "max_drawdown_soft")
    ensure_range(max_dd_soft, -0.5, -0.01, "max_drawdown_soft")

    position_max_gross = get_float(risk_config, "position_max_gross")
    ensure_range(position_max_gross, 0.0, 1.5, "position_max_gross")
    if position_max_gross <= 0:
        raise ValidationError("position_max_gross must be greater than 0")

    position_max_per_asset = get_float(risk_config, "position_max_per_asset")
    ensure_range(position_max_per_asset, 0.0, 1.0, "position_max_per_asset")
    if position_max_per_asset <= 0:
        raise ValidationError("position_max_per_asset must be greater than 0")

    risk_per_trade = get_float(risk_config, "risk_per_trade")
    ensure_range(risk_per_trade, 0.0, 0.05, "risk_per_trade")
    if risk_per_trade <= 0:
        raise ValidationError("risk_per_trade must be greater than 0")

    daily_loss_limit = get_float(risk_config, "daily_loss_limit")
    ensure_range(daily_loss_limit, -0.50, -0.001, "daily_loss_limit")

    cooldown_days_after_hit = get_float(risk_config, "cooldown_days_after_hit")
    ensure_range(cooldown_days_after_hit, 0.0, 10.0, "cooldown_days_after_hit")

    slippage_bps = get_float(risk_config, "slippage_bps")
    ensure_range(slippage_bps, 0.0, 100.0, "slippage_bps")

    commission_per_share = get_float(risk_config, "commission_per_share")
    ensure_range(commission_per_share, 0.0, 0.05, "commission_per_share")

    stop_atr_mult = get_float(risk_config, "stop_atr_mult")
    if stop_atr_mult <= 0:
        raise ValidationError("stop_atr_mult must be positive")

    takeprofit_rr = get_float(risk_config, "takeprofit_rr")
    if takeprofit_rr <= 0:
        raise ValidationError("takeprofit_rr must be positive")


def validate_backtest_config(backtest_config: dict[str, Any]) -> None:
    required_keys = [
        "start",
        "end",
        "cash",
        "include_costs",
        "apply_t_plus_one",
        "walk_forward",
    ]
    require_keys(backtest_config, required_keys, "backtest.yml")

    walk_forward = backtest_config.get("walk_forward")
    if not isinstance(walk_forward, dict):
        raise ValidationError("walk_forward must be a mapping with training windows")

    wf_required = ["window_train_days", "window_test_days", "step_days"]
    require_keys(walk_forward, wf_required, "backtest.yml walk_forward")

    try:
        cash = float(backtest_config["cash"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValidationError("cash must be a numeric value greater than 0") from exc
    if cash <= 0:
        raise ValidationError("cash must be a numeric value greater than 0")


def main() -> None:
    try:
        risk_config = load_yaml(RISK_CONFIG_PATH)
        validate_risk_config(risk_config)

        backtest_config = load_yaml(BACKTEST_CONFIG_PATH)
        validate_backtest_config(backtest_config)
    except ValidationError as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    print("Config validation passed ✅")


if __name__ == "__main__":
    main()
