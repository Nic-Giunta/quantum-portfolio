import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

class ExperimentConfig(BaseModel):
    name: str
    seed: int | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    def config_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]
    def save_json(self, path): Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

@dataclass
class ExperimentResult:
    config: ExperimentConfig
    metrics: dict[str, float]
    artifacts: dict[str, str] = field(default_factory=dict)
