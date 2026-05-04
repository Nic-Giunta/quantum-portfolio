from quantum_portfolio.config import load_optimization_config
from quantum_portfolio.data.synthetic import make_synthetic_returns


def test_config(tmp_path):
    r = make_synthetic_returns(n_periods=20); p = tmp_path/"r.csv"; r.to_csv(p)
    cfg = tmp_path/"c.json"; cfg.write_text('{"returns_path": "' + str(p) + '", "constraints": [{"name": "LongOnly", "params": {}}]}')
    assert load_optimization_config(cfg).returns_path == str(p)
