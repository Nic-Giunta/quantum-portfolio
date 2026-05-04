from .base import ExpectedReturnModel
from .historical import HistoricalMean, ExponentialWeightedMean
from .shrinkage import JamesSteinShrinkage, BayesianShrinkage
from .factor import CAPMExpectedReturns, FactorExpectedReturns, PCAFactorExpectedReturns
from .black_litterman import BlackLitterman
from .ensembles import EnsembleExpectedReturns
from .ml import FeatureBuilder, MLExpectedReturns, WalkForwardForecaster, ModelSignalAllocator
