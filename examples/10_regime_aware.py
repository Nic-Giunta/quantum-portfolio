from quantum_portfolio.data.synthetic import make_synthetic_returns
from quantum_portfolio.regimes import VolatilityRegimeDetector
print(VolatilityRegimeDetector(window=20).fit_predict(make_synthetic_returns(seed=16)).tail())
