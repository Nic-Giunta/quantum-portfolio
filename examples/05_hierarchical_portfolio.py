from quantum_portfolio.data.synthetic import make_synthetic_returns
from quantum_portfolio.hierarchical import HRPAllocator
print(HRPAllocator().allocate(make_synthetic_returns(seed=11)))
