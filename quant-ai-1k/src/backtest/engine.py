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
        rebalance=str(backtest_data.get("rebalance", universe_data.get("rebalance", "weekly"))).lower(),
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

        if end is not None:
            closes = closes[(closes.index >= start) & (closes.index <= end)]
            highs = highs[(highs.index >= start) & (highs.index <= end)]
            lows = lows[(lows.index >= start) & (lows.index <= end)]
        else:
            closes = closes[closes.index >= start]
            highs = highs[highs.index >= start]
            lows = lows[lows.index >= start]

        closes = closes.dropna(how="all")
        highs = highs.reindex(closes.index).ffill()
        lows = lows.reindex(closes.index).ffill()

        if closes.empty:
            raise ValueError(
                f"No data after {start.strftime('%Y-%m-%d')}. Check your start_date in config/backtest.yml"
            )

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
                        base_weight = min(1.0 / n, self.risk_config.max_position_weight)
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

        gross_returns = pd.Series(0.0, index=closes.index)
        net_returns = pd.Series(0.0, index=closes.index)
        turnover = pd.Series(0.0, index=closes.index)
        slippage_cost = pd.Series(0.0, index=closes.index)
        fees = pd.Series(0.0, index=closes.index)
        cooldown_on = pd.Series(False, index=closes.index)
        effective_weights = pd.DataFrame(0.0, index=closes.index, columns=self.symbols)

        prev_positions = pd.Series(0.0, index=self.symbols)
        prev_equity_value = self.config.initial_cash
        cooldown_remaining = 0

        for idx, date in enumerate(closes.index):
            if self.config.t_plus_one:
                base_weights = weights.iloc[idx - 1] if idx > 0 else pd.Series(0.0, index=self.symbols)
            else:
                base_weights = weights.iloc[idx]
            base_weights = base_weights.fillna(0.0)

            cooldown_active = cooldown_remaining > 0
            if cooldown_active:
                current_effective = pd.Series(0.0, index=self.symbols)
            else:
                current_effective = base_weights.copy()

            effective_weights.loc[date] = current_effective
            cooldown_on.loc[date] = cooldown_active

            daily_turnover = (current_effective - prev_positions).abs().sum()
            turnover.loc[date] = daily_turnover
            slippage_cost.loc[date] = daily_turnover * (self.config.slippage_bps / 10000.0)

            commission_value = 0.0
            if rebalance_mask.loc[date]:
                previous_target = weights.iloc[idx - 1] if idx > 0 else pd.Series(0.0, index=self.symbols)
                delta_weights = (weights.iloc[idx] - previous_target).abs().fillna(0.0)
                if delta_weights.sum() > 0 and prev_equity_value > 0:
                    prices = closes.loc[date]
                    for sym in self.symbols:
                        price = prices.get(sym)
                        if pd.isna(price) or price <= 0:
                            continue
                        change = float(delta_weights.get(sym, 0.0))
                        if change <= 0:
                            continue
                        shares = int(np.floor(prev_equity_value * change / price))
                        if shares <= 0:
                            continue
                        commission_value += shares * self.config.commission_per_trade
            commission_cost_return = commission_value / prev_equity_value if prev_equity_value > 0 else 0.0

            daily_gross = float((current_effective * returns.loc[date].fillna(0.0)).sum())
            gross_returns.loc[date] = daily_gross

            total_fees = slippage_cost.loc[date] + commission_cost_return
            fees.loc[date] = total_fees
            daily_net = daily_gross - total_fees
            net_returns.loc[date] = daily_net

            prev_equity_value *= (1 + daily_net)
            prev_positions = current_effective

            next_cooldown = cooldown_remaining - 1 if cooldown_active else cooldown_remaining
            next_cooldown = max(next_cooldown, 0)
            if daily_net <= -self.risk_config.daily_loss_cooldown:
                next_cooldown = self.risk_config.cooldown_days
            cooldown_remaining = next_cooldown

        equity_curve = (1 + net_returns).cumprod()
        equity_value = equity_curve * self.config.initial_cash

        rolling_max = equity_curve.cummax()
        drawdown = equity_curve / rolling_max - 1.0
        max_drawdown = drawdown.min()

        annual_factor = np.sqrt(252)
        avg_return = net_returns.mean()
        vol = net_returns.std()
        ann_vol = vol * annual_factor
        sharpe = (avg_return / vol * annual_factor) if vol > 0 else np.nan

        downside = net_returns[net_returns < 0]
        downside_std = downside.std()
        if pd.notna(downside_std) and downside_std > 0:
            sortino = (avg_return / downside_std) * annual_factor
        else:
            sortino = float('nan')

        trading_days = len(net_returns)
        ending_equity = equity_curve.iloc[-1] if trading_days > 0 else np.nan
        if trading_days > 0 and ending_equity > 0:
            cagr = float(ending_equity ** (252 / trading_days) - 1)
        else:
            cagr = float('nan')
        if not np.isnan(cagr) and not np.isnan(max_drawdown) and abs(max_drawdown) > 0:
            calmar = float(cagr / abs(max_drawdown))
        else:
            calmar = float('nan')

        summary = {
            "total_return": float(ending_equity - 1.0) if not np.isnan(ending_equity) else float('nan'),
            "max_drawdown": float(max_drawdown),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "sortino": float(sortino) if not pd.isna(sortino) else float("nan"),
            "cagr": cagr,
            "calmar": calmar,
            "trading_days": int(trading_days),
        }

        timeseries = pd.DataFrame(
            {
                "equity_curve": equity_curve,
                "equity_value": equity_value,
                "gross_return": gross_returns,
                "net_return": net_returns,
                "turnover": turnover,
                "fees": fees,
                "drawdown": drawdown,
                "cooldown_on": cooldown_on,
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
