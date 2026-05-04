from quantum_portfolio.hierarchical import HRPAllocator
def test_hrp(returns):
    w = HRPAllocator().allocate(returns); assert abs(w.sum()-1) < 1e-8 and (w >= 0).all()
