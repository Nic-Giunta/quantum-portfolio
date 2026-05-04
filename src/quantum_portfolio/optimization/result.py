from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
from quantum_portfolio.risk.risk_budgeting import component_risk_contribution
from quantum_portfolio.version import __version__

@dataclass
class OptimizationResult:
    weights: pd.Series
    expected_return: float
    risk: float
    objective_value: float | None
    solver_status: str
    solver_name: str
    solve_time: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    constraints_report: list[dict[str, Any]] = field(default_factory=list)
    risk_contributions: pd.Series | None = None
    factor_exposures: pd.Series | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    library_version: str = __version__
    config_hash: str | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_weights(cls, weights, expected_returns, covariance, *, objective_value, solver_status, solver_name, solve_time, diagnostics=None, constraints_report=None, warnings=None):
        er = float(weights @ expected_returns.reindex(weights.index))
        cov = covariance.loc[weights.index, weights.index]
        risk = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
        rc = component_risk_contribution(weights, cov)
        return cls(weights, er, risk, objective_value, solver_status, solver_name, solve_time, diagnostics or {}, constraints_report or [], rc, warnings=warnings or [])

    def summary(self) -> str:
        sharpe = self.expected_return / self.risk if self.risk > 1e-12 else np.nan
        return "\n".join([
            "OptimizationResult",
            f"  status: {self.solver_status}",
            f"  solver: {self.solver_name}",
            f"  expected_return: {self.expected_return:.6f}",
            f"  risk: {self.risk:.6f}",
            f"  sharpe_like: {sharpe:.6f}",
            f"  n_assets: {len(self.weights)}",
        ])

    def to_frame(self) -> pd.DataFrame:
        df = self.weights.rename("weight").to_frame()
        if self.risk_contributions is not None: df["risk_contribution"] = self.risk_contributions.reindex(df.index)
        if self.factor_exposures is not None: df["factor_exposure"] = self.factor_exposures.reindex(df.index)
        return df

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.to_dict(),
            "expected_return": self.expected_return,
            "risk": self.risk,
            "objective_value": self.objective_value,
            "solver_status": self.solver_status,
            "solver_name": self.solver_name,
            "solve_time": self.solve_time,
            "diagnostics": self.diagnostics,
            "constraints_report": self.constraints_report,
            "risk_contributions": None if self.risk_contributions is None else self.risk_contributions.to_dict(),
            "timestamp": self.timestamp,
            "library_version": self.library_version,
            "config_hash": self.config_hash,
            "warnings": self.warnings,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
    def save_weights_csv(self, path: str | Path) -> None: self.to_frame().to_csv(path)
    def plot_weights(self):
        ax = self.weights.sort_values().plot(kind="bar", title="Portfolio weights"); ax.set_ylabel("Weight"); return ax
    def plot_risk_contributions(self):
        rc = self.risk_contributions if self.risk_contributions is not None else pd.Series(dtype=float)
        ax = rc.sort_values().plot(kind="bar", title="Risk contributions"); ax.set_ylabel("Contribution"); return ax
