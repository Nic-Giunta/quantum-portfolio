from quantum_portfolio.data.synthetic import make_synthetic_returns
from quantum_portfolio.scenarios import ShockScenario, ScenarioAnalyzer
r = make_synthetic_returns(seed=18); w = r.iloc[-1]*0 + 1/r.shape[1]
print(ScenarioAnalyzer(w).report(ShockScenario(parallel_shock=-.02).apply(r.tail(10))))
