import json
from pathlib import Path


class LocalExperimentRegistry:
    def __init__(self, root="runs"):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
    def save(self, result, weights=None):
        d = self.root / f"{result.config.name}-{result.config.config_hash()}"; d.mkdir(parents=True, exist_ok=True)
        result.config.save_json(d/"config.json"); (d/"metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
        if weights is not None: weights.to_csv(d/"weights.csv")
        return d
