import re

from typer.testing import CliRunner

from ganyan.cli.main import app

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(result):
    # Rich colours help output when it detects CI (GITHUB_ACTIONS), which
    # splits "--today" across escape codes; compare on plain text.
    return _ANSI.sub("", result.output)


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ganyan" in result.output.lower() or "scrape" in result.output.lower()


def test_scrape_help():
    result = runner.invoke(app, ["scrape", "--help"])
    assert result.exit_code == 0
    assert "--today" in _plain(result)


def test_predict_help():
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0


def test_races_help():
    result = runner.invoke(app, ["races", "--help"])
    assert result.exit_code == 0


def test_evaluate_help():
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--detail" in _plain(result)
    assert "--json" in _plain(result)


def test_db_help():
    result = runner.invoke(app, ["db", "--help"])
    assert result.exit_code == 0
    assert "init" in result.output


# _format_race_header — tests the predict-output upgrade (PR #3 idea
# implemented locally without the SQL view + migration).


class _FakeTrack:
    def __init__(self, name):
        self.name = name


class _FakeRace:
    def __init__(self, **kw):
        from datetime import date as _date
        self.track = _FakeTrack(kw.get("track", "İzmir"))
        self.race_number = kw.get("race_number", 8)
        self.date = kw.get("date", _date(2026, 5, 7))
        self.post_time = kw.get("post_time", "21:30")
        self.distance_meters = kw.get("distance_meters", 1100)
        self.surface = kw.get("surface", "kum")
        self.race_type = kw.get("race_type", "ŞARTLI 19/DHÖW")
        self.horse_type = kw.get("horse_type", "3 Yaşlı Araplar")


def test_format_race_header_full():
    from ganyan.cli.main import _format_race_header
    line1, line2 = _format_race_header(_FakeRace())
    assert line1 == "İzmir (1100 m, kum) - #8 (07.05.2026 @ 21:30)"
    assert line2 == "3 Yaşlı Araplar, ŞARTLI 19/DHÖW"


def test_format_race_header_handles_missing_post_time():
    from ganyan.cli.main import _format_race_header
    line1, _ = _format_race_header(_FakeRace(post_time=None))
    assert "@" not in line1
    assert "07.05.2026" in line1


def test_format_race_header_handles_missing_distance_and_surface():
    from ganyan.cli.main import _format_race_header
    line1, _ = _format_race_header(_FakeRace(distance_meters=None, surface=None))
    # Parens block dropped entirely when both bits are missing.
    assert "(" not in line1.split(" - ")[0]


def test_format_race_header_returns_empty_for_none():
    from ganyan.cli.main import _format_race_header
    assert _format_race_header(None) == ("", "")
