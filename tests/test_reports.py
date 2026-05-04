from quantum_portfolio import PortfolioOptimizer
from quantum_portfolio.constraints import LongOnly
from quantum_portfolio.objectives import MinVariance
from quantum_portfolio.reports import html_report, markdown_report


def test_reports(returns):
    r = PortfolioOptimizer(returns, objective=MinVariance(), constraints=[LongOnly()]).solve()
    assert "Optimization Report" in markdown_report(r)
    assert "<html" in html_report(r)
