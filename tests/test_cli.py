from typer.testing import CliRunner
from quantum_portfolio.cli.app import app
def test_cli_version():
    res = CliRunner().invoke(app, ["version"])
    assert res.exit_code == 0
