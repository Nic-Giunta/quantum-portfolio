from dataclasses import dataclass, field


@dataclass
class PluginRegistry:
    risk_models: dict = field(default_factory=dict)
    expected_return_models: dict = field(default_factory=dict)
    objectives: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    def register_risk_model(self, name, cls): self.risk_models[name]=cls
    def register_expected_return_model(self, name, cls): self.expected_return_models[name]=cls
    def register_objective(self, name, cls): self.objectives[name]=cls
    def register_constraint(self, name, cls): self.constraints[name]=cls
