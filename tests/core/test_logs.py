import re
from collections.abc import Iterator

import pytest
import structlog

from policies.core.logs import configure, get_logger

Capture = pytest.CaptureFixture[str]


@pytest.fixture(autouse=True)
def unconfigured() -> Iterator[None]:
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def test_unconfigured_loggers_show_info_and_up_on_stderr(capsys: Capture) -> None:
    log = get_logger("policies.test")
    log.debug("plugin loaded")
    log.info("checkpoint", step=3)
    log.bind(run="a").warning("hook disabled")
    captured = capsys.readouterr()
    assert captured.out == ""
    first, second = captured.err.splitlines()
    assert re.match(r"\d\d:\d\d:\d\dZ \[info\s*\] checkpoint step=3$", first)
    assert "hook disabled" in second and "run=a" in second
    assert not structlog.is_configured()


def test_configuring_later_takes_over(capsys: Capture) -> None:
    log = get_logger("policies.test")
    configure("debug")
    log.debug("plugin loaded")
    assert "plugin loaded" in capsys.readouterr().err
    with structlog.testing.capture_logs() as logs:
        log.bind(run="a").debug("fit epoch", epoch=1)
    assert logs == [{"event": "fit epoch", "log_level": "debug", "run": "a", "epoch": 1}]


@pytest.mark.parametrize("level", ["warning", "Warning", "WARNING", 30])
def test_levels_in_any_case(level: str | int, capsys: Capture) -> None:
    configure(level)
    log = get_logger("policies.test")
    log.info("hidden")
    log.warning("shown")
    assert capsys.readouterr().err.count("\n") == 1


def test_unknown_level() -> None:
    with pytest.raises(ValueError, match="unknown log level 'loud'"):
        configure("loud")


def test_json_lines_have_utc_iso_times(capsys: Capture) -> None:
    configure("info", json=True)
    get_logger("policies.test").info("checkpoint")
    assert re.search(r'"timestamp": "[^"]+Z"', capsys.readouterr().err)
