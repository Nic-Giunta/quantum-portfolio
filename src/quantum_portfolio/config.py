import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from quantum_portfolio.constraints import LongOnly, MaxWeight, WeightSum
from quantum_portfolio.expected_returns import ExponentialWeightedMean, HistoricalMean
from quantum_portfolio.objectives import MeanVarianceUtility, MinVariance
from quantum_portfolio.risk import LedoitWolfCovariance, SampleCovariance
from quantum_portfolio.utils.exceptions import ConfigurationError

_WINDOWS_PATH_BACKSLASH = re.compile(r'(?<!\\)\\(?=[A-Za-z0-9_. -])')

class ComponentConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)

class OptimizationConfig(BaseModel):
    returns_path: str
    expected_return_model: ComponentConfig = ComponentConfig(name="HistoricalMean")
    risk_model: ComponentConfig = ComponentConfig(name="SampleCovariance")
    objective: ComponentConfig = ComponentConfig(name="MinVariance")
    constraints: list[ComponentConfig] = Field(default_factory=lambda: [ComponentConfig(name="LongOnly")])
    solver: str | None = None

def _load_raw(path):
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ConfigurationError("YAML requires PyYAML; install [cli].") from exc
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        if "Invalid \\escape" not in str(exc):
            raise
        return json.loads(_WINDOWS_PATH_BACKSLASH.sub(r"\\\\", text))

def load_optimization_config(path) -> OptimizationConfig: return OptimizationConfig.model_validate(_load_raw(path))

def build_component(config: ComponentConfig):
    mapping = {"HistoricalMean": HistoricalMean, "ExponentialWeightedMean": ExponentialWeightedMean, "SampleCovariance": SampleCovariance, "LedoitWolfCovariance": LedoitWolfCovariance, "MinVariance": MinVariance, "MeanVarianceUtility": MeanVarianceUtility, "LongOnly": LongOnly, "MaxWeight": MaxWeight, "WeightSum": WeightSum}
    if config.name not in mapping: raise ConfigurationError(f"Unknown component {config.name}")
    return mapping[config.name](**config.params)
