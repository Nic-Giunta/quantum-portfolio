from pathlib import Path

from quantum_portfolio import __version__
from quantum_portfolio.config import build_component, load_optimization_config
from quantum_portfolio.data.loaders import load_csv_timeseries
from quantum_portfolio.data.validators import validate_returns_dataframe
from quantum_portfolio.optimization import PortfolioOptimizer

try:
    import typer
except ImportError as exc:
    raise ImportError("CLI requires quantum-portfolio[cli].") from exc

app = typer.Typer(help="QuantumPortfolio CLI")

@app.command("version")
def version(): typer.echo(__version__)

@app.command("validate-data")
def validate_data(path: Path):
    typer.echo(validate_returns_dataframe(load_csv_timeseries(path), allow_missing=True).to_dict())

@app.command("optimize")
def optimize(config_path: Path):
    cfg = load_optimization_config(config_path); returns = load_csv_timeseries(cfg.returns_path)
    opt = PortfolioOptimizer(returns, build_component(cfg.expected_return_model), build_component(cfg.risk_model), build_component(cfg.objective), [build_component(c) for c in cfg.constraints])
    res = opt.solve(solver=cfg.solver); typer.echo(res.summary()); res.save_json("optimization_result.json")

@app.command("backtest")
def backtest(config_path: Path): typer.echo(f"Backtest config accepted: {config_path}. Use Python API for custom strategy injection.")

@app.command("report")
def report(results_json: Path):
    import json
    data = json.loads(results_json.read_text(encoding="utf-8"))
    typer.echo(f"Loaded result status={data.get('solver_status')} weights={len(data.get('weights', {}))}")

def main(): app()
