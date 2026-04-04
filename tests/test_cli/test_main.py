from typer.testing import CliRunner

from ganyan.cli.main import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ganyan" in result.output.lower() or "scrape" in result.output.lower()


def test_scrape_help():
    result = runner.invoke(app, ["scrape", "--help"])
    assert result.exit_code == 0
    assert "--today" in result.output


def test_predict_help():
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0


def test_races_help():
    result = runner.invoke(app, ["races", "--help"])
    assert result.exit_code == 0


def test_db_help():
    result = runner.invoke(app, ["db", "--help"])
    assert result.exit_code == 0
    assert "init" in result.output
