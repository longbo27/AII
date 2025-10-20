"""Risk configuration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RiskConfig:
    """Dataclass encapsulating risk management parameters."""

    max_drawdown_hard: float
    max_drawdown_soft: float
    max_position_weight: float
    max_trade_risk: float
    daily_loss_cooldown: float
    cooldown_days: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskConfig":
        """Instantiate :class:`RiskConfig` from a mapping."""
        return cls(
            max_drawdown_hard=float(data["max_drawdown_hard"]),
            max_drawdown_soft=float(data["max_drawdown_soft"]),
            max_position_weight=float(data["max_position_weight"]),
            max_trade_risk=float(data["max_trade_risk"]),
            daily_loss_cooldown=float(data["daily_loss_cooldown"]),
            cooldown_days=int(data["cooldown_days"]),
        )


def load_risk_config(path: str | Path) -> RiskConfig:
    """Load the risk configuration from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("Risk configuration must be a mapping")
    return RiskConfig.from_dict(data)


__all__ = ["RiskConfig", "load_risk_config"]
