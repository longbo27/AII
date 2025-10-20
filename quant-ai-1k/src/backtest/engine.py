"""Simple vectorized backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from ..core.features import atr_14
from ..utils.risk import RiskConfig


@dataclass(slots=True)
class BacktestConfig:
    """Container for backtest-related parameters."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp | None
    initial_cash: float
    commission_per_trade: float
    slippage_bps: float
    t_plus_one: bool
    rebalance: str
    equal_weight: bool


@dataclass(slots=True)
class UniverseConfig:
    """Universe-level configuration."""

    symbols: list[str]
    cash_symbol: str


def load_backtest_config(path: str | Path, universe_path: str | Path) -> tuple[BacktestConfig, UniverseConfig]:
    """Load backtest and universe configuration from YAML files."""
    with Path(path).open("r", encoding="utf-8") as fh:
        backtest_data = yaml.safe_load(fh)
    with Path(universe_path).open("r", encoding="utf-8") as fh:
        universe_data = yaml.safe_load(fh)

    if not isinstance(backtest_data, dict):
        raise ValueError("Backtest configuration must be a mapping")
    if not isinstance(universe_data, dict):
        raise ValueError("Universe configuration must be a mapping")

    config = BacktestConfig(
        start_date=pd.to_datetime(backtest_data["start_date"]),
        end_date=pd.to_datetime(backtest_data.get("end_date")) if backtest_data.get("end_date") else None,
        initial_cash=float(backtest_data["initial_cash"]),
        commission_per_trade=float(backtest_data.get("commission_per_trade", 0.0)),
        slippage_bps=float(backtest_data.get("slippage_bps", 0.0)),
        t_plus_one=bool(backtest_data.get("t_plus_one", True)),
        rebalance=str(universe_data.get("rebalance", "weekly")).lower(),
        equal_weight=bool(universe_data.get("equal_weight", True)),
    )

    universe = UniverseConfig(
        symbols=[str(symbol) for symbol in universe_data.get("symbols", [])],
        cash_symbol=str(universe_data.get("cash_symbol", "CASH")),
    )

    return config, universe


class BacktestEngine:
    """Implements a minimal long-only vectorized backtest."""

    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        config: BacktestConfig,
        risk_config: RiskConfig,
    ) -> None:
        self.price_data = price_data
        self.config = config
        self.risk_config = risk_config
        self.symbols = list(price_data.keys())

    def _prepare_panels(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        closes = pd.DataFrame({sym: df["Close"] for sym, df in self.price_data.items()})
        highs = pd.DataFrame({sym: df["High"] for sym, df in self.price_data.items()})
        lows = pd.DataFrame({sym: df["Low"] for sym, df in self.price_data.items()})

        closes = closes.sort_index().ffill()
        highs = highs.sort_index().ffill()
        lows = lows.sort_index().ffill()

        start = self.config.start_date
        end = self.config.end_date
        if end:
            closes = closes.loc[start:end]
            highs = highs.loc[start:end]
            lows = lows.loc[start:end]
        else:
            closes = closes.loc[start:]
            highs = highs.loc[start:]
            lows = lows.loc[start:]

        closes = closes.dropna(how="all")
        highs = highs.reindex(closes.index).ffill()
        lows = lows.reindex(closes.index).ffill()

        return closes, highs, lows

    def _rebalance_mask(self, index: pd.Index) -> pd.Series:
        mask = pd.Series(False, index=index)
        rebalance = self.config.rebalance
        if rebalance == "weekly":
            mask.loc[index[index.weekday == 0]] = True
        elif rebalance == "monthly":
            mask.loc[index[index.is_month_start]] = True
        else:  # daily or unsupported defaults to daily
            mask[:] = True
        if not mask.iloc[0]:
            mask.iloc[0] = True
        return mask

    def run(self) -> dict[str, object]:
        """Execute the backtest and return results."""
        closes, highs, lows = self._prepare_panels()
        if closes.empty:
            raise ValueError("No price data available for the requested date range")

        logger.info("Running backtest on {} symbols", len(self.symbols))

        atr_values = {sym: atr_14(highs[sym], lows[sym], closes[sym]) for sym in self.symbols}
        atr_df = pd.DataFrame(atr_values)
        atr_ratio = atr_df / closes

        sma20 = closes.rolling(window=20, min_periods=20).mean()
        sma50 = closes.rolling(window=50, min_periods=50).mean()
        trend_filter = sma20 > sma50
        vol_filter = atr_ratio < 0.10
        eligibility = trend_filter & vol_filter

        weights = pd.DataFrame(0.0, index=closes.index, columns=self.symbols)
        rebalance_mask = self._rebalance_mask(closes.index)
        previous_weights = pd.Series(0.0, index=self.symbols)

        for date in closes.index:
            if rebalance_mask.loc[date]:
                eligible = eligibility.loc[date].fillna(False)
                target = pd.Series(0.0, index=self.symbols)
                eligible_symbols = eligible[eligible].index.tolist()
                if eligible_symbols:
                    n = len(eligible_symbols)
                    if self.config.equal_weight:
                        base_weight = 1.0 / n
                        base_weight = min(base_weight, self.risk_config.max_position_weight)
                        target.loc[eligible_symbols] = base_weight
                        total_weight = target.sum()
                        if total_weight > 1.0:
                            target *= 1.0 / total_weight
                    else:
                        target.loc[eligible_symbols] = min(
                            1.0 / n, self.risk_config.max_position_weight
                        )
                previous_weights = target
            weights.loc[date] = previous_weights

        returns = closes.pct_change().fillna(0.0)
        effective_weights = weights.shift(1 if self.config.t_plus_one else 0).fillna(0.0)

        gross_returns = (effective_weights * returns).sum(axis=1)
        turnover = weights.diff().abs().fillna(weights.iloc[0]).sum(axis=1)
        slippage_cost = turnover * (self.config.slippage_bps / 10000.0)
        net_returns = gross_returns - slippage_cost

        equity_curve = (1 + net_returns).cumprod()
        equity_value = equity_curve * self.config.initial_cash

        rolling_max = equity_curve.cummax()
        drawdown = equity_curve / rolling_max - 1.0
        max_drawdown = drawdown.min()

        annual_factor = np.sqrt(252)
        avg_return = net_returns.mean()
        vol = net_returns.std()
        sharpe = (avg_return / vol * annual_factor) if vol > 0 else np.nan

        summary = {
            "total_return": float(equity_curve.iloc[-1] - 1.0),
            "max_drawdown": float(max_drawdown),
            "daily_vol": float(vol * annual_factor),
            "sharpe": float(sharpe),
            "trading_days": int(len(net_returns)),
        }

        timeseries = pd.DataFrame(
            {
                "equity_curve": equity_curve,
                "equity_value": equity_value,
                "gross_return": gross_returns,
                "net_return": net_returns,
                "turnover": turnover,
                "drawdown": drawdown,
            }
        )

        logger.info(
            "Backtest completed: total_return={:.2%}, max_drawdown={:.2%}, sharpe={:.2f}",
            summary["total_return"],
            summary["max_drawdown"],
            summary["sharpe"],
        )

        return {
            "summary": summary,
            "timeseries": timeseries,
            "weights": weights,
            "eligibility": eligibility,
        }


__all__ = [
    "BacktestConfig",
    "UniverseConfig",
    "load_backtest_config",
    "BacktestEngine",
]
