import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import structlog

from policies.core.logs import configure
from policies.training.monitoring import (
    Event,
    EventTraits,
    FunctionHook,
    HookList,
    build_hooks,
    event_traits,
    hooks,
)


class Recorder:
    def __init__(self, every: int = 1) -> None:
        self.every = every
        self.events: list[Event] = []
        self.closed = False

    def __call__(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


class Broken:
    every = 1

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, event: Event) -> None:
        self.calls += 1
        raise RuntimeError("boom")


def test_builtins_are_registered() -> None:
    assert {"console", "jsonl", "tensorboard", "wandb"} <= set(hooks.names())


def test_every_applies_to_iteration_and_update_only() -> None:
    recorder = Recorder(every=10)
    hook_list = HookList([recorder])
    for step in range(25):
        hook_list.emit(Event("iteration", step))
        hook_list.emit(Event("update", step))
    hook_list.emit(Event("evaluation", 3))
    hook_list.emit(Event("checkpoint", 7))
    kinds = [(event.kind, event.step) for event in recorder.events]
    assert kinds == [
        ("iteration", 0),
        ("update", 0),
        ("iteration", 10),
        ("update", 10),
        ("iteration", 20),
        ("update", 20),
        ("evaluation", 3),
        ("checkpoint", 7),
    ]


def test_custom_kinds_opt_in_to_throttling_and_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(event_traits, "tests.probe", EventTraits(throttled=True, scalar=True))
    recorder = Recorder(every=10)
    hook_list = HookList([recorder])
    for step in range(25):
        hook_list.emit(Event("tests.probe", step))
        hook_list.emit(Event("tests.other", step))
    probes = [event.step for event in recorder.events if event.kind == "tests.probe"]
    assert probes == [0, 10, 20]
    assert len(recorder.events) == 3 + 25
    console = build_hooks([{"type": "console", "every": 1, "max_metrics": 1}], tmp_path)
    with structlog.testing.capture_logs() as logs:
        console.emit(Event("tests.probe", 1, {"score": 0.5, "note": "x"}))
    assert logs == [{"event": "tests.probe", "log_level": "info", "step": 1, "score": 0.5}]


def test_failing_hook_is_logged_once_and_disabled() -> None:
    broken, recorder = Broken(), Recorder()
    hook_list = HookList({"broken": broken, "good": recorder})
    with structlog.testing.capture_logs() as logs:
        for step in range(3):
            hook_list.emit(Event("iteration", step))
    assert [(log["event"], log["hook"]) for log in logs] == [("hook disabled", "broken")]
    assert broken.calls == 1
    assert len(recorder.events) == 3
    assert hook_list.names == ["good"]


def test_a_hook_may_emit_while_a_failing_hook_is_disabled() -> None:
    def echo(event: Event) -> None:
        if event.kind == "checkpoint":
            hook_list.emit(Event("evaluation", event.step))

    broken, recorder = Broken(), Recorder()
    hook_list = HookList({"echo": FunctionHook(echo), "record": recorder, "broken": broken})
    with structlog.testing.capture_logs() as logs:
        hook_list.emit(Event("checkpoint", 1))
    assert broken.calls == 1 and len(logs) == 1
    assert [event.kind for event in recorder.events] == ["evaluation", "checkpoint"]


def test_close_calls_hooks_that_have_close() -> None:
    recorder = Recorder()
    hook_list = HookList([recorder, Broken()])
    hook_list.close()
    assert recorder.closed


def test_jsonl_writes_one_line_per_event(tmp_path: Path) -> None:
    hook_list = build_hooks([{"type": "jsonl"}], tmp_path)
    hook_list.emit(Event("start", 0, {"seed": 1}))
    hook_list.emit(Event("iteration", 1, {"reward": 0.5, "path": Path("x")}))
    hook_list.close()
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [record["kind"] for record in records] == ["start", "iteration"]
    assert records[1]["data"] == {"reward": 0.5, "path": "x"}
    assert isinstance(records[0]["time"], float)


def test_jsonl_appends_and_flushes(tmp_path: Path) -> None:
    hook_list = build_hooks([{"type": "jsonl", "path": "logs/events.jsonl"}], tmp_path)
    hook_list.emit(Event("start", 0))
    assert (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").count("\n") == 1
    hook_list.close()
    again = build_hooks([{"type": "jsonl", "path": "logs/events.jsonl"}], tmp_path)
    again.emit(Event("end", 5))
    again.close()
    assert (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").count("\n") == 2


def test_console_logs_first_scalars_of_metric_events(tmp_path: Path) -> None:
    hook_list = build_hooks([{"type": "console", "every": 2, "max_metrics": 2}], tmp_path)
    data = {"reward": 1.23456, "loss": {"value": 0.5}, "length": 7, "note": "x"}
    with structlog.testing.capture_logs() as logs:
        hook_list.emit(Event("iteration", 1, data))
        hook_list.emit(Event("iteration", 2, data))
        hook_list.emit(Event("checkpoint", 3, {"path": "a/b.pt"}))
        hook_list.emit(Event("error", 3, {"error": "bad"}))
    assert logs == [
        {"event": "iteration", "log_level": "info", "step": 2, "reward": 1.235, "loss/value": 0.5},
        {"event": "checkpoint", "log_level": "info", "step": 3, "path": "a/b.pt"},
        {"event": "error", "log_level": "error", "step": 3, "error": "bad"},
    ]


def test_console_renders_one_line_on_stderr_once_configured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure()
    try:
        hook_list = build_hooks([{"type": "console"}], tmp_path)
        hook_list.emit(Event("evaluation", 5, {"reward": 2.5}))
    finally:
        structlog.reset_defaults()
    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert re.search(r"\binfo\b.*\bevaluation\b", line)
    assert "reward=2.5" in line
    assert "step=5" in line


def test_tensorboard_logs_finite_scalars(tmp_path: Path) -> None:
    pytest.importorskip("torch.utils.tensorboard")
    accumulator = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")
    hook_list = build_hooks([{"type": "tensorboard"}], tmp_path)
    hook_list.emit(Event("iteration", 1, {"reward": 1.5, "loss": {"value": math.nan}, "note": "x"}))
    hook_list.emit(Event("evaluation", 2, {"reward": 3.0}))
    hook_list.emit(Event("checkpoint", 3, {"size": 9}))
    hook_list.close()
    events = accumulator.EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()
    assert sorted(events.Tags()["scalars"]) == ["evaluation/reward", "iteration/reward"]
    [scalar] = events.Scalars("iteration/reward")
    assert (scalar.step, scalar.value) == (1, 1.5)


def test_tensorboard_without_the_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    with pytest.raises(ImportError, match="tensorboard hook needs"):
        build_hooks([{"type": "tensorboard"}], tmp_path)


class FakeRun:
    def __init__(self, **settings: Any) -> None:
        self.settings = settings
        self.logged: list[tuple[dict[str, float], int]] = []
        self.finished = False

    def log(self, values: dict[str, float], step: int) -> None:
        self.logged.append((values, step))

    def finish(self) -> None:
        self.finished = True


def test_wandb_logs_scalars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[FakeRun] = []

    def init(**settings: Any) -> FakeRun:
        runs.append(FakeRun(**settings))
        return runs[-1]

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    hook_list = build_hooks([{"type": "wandb", "project": "walk", "name": "first"}], tmp_path)
    hook_list.emit(Event("update", 1, {"loss": {"value": 0.5}, "note": "x"}))
    hook_list.emit(Event("iteration", 2, {"note": "x"}))
    hook_list.emit(Event("checkpoint", 3, {"size": 9}))
    hook_list.close()
    [run] = runs
    assert run.settings == {
        "project": "walk",
        "entity": None,
        "name": "first",
        "dir": str(tmp_path),
    }
    assert run.logged == [({"update/loss/value": 0.5}, 1)]
    assert run.finished


def test_wandb_without_the_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match="wandb hook needs"):
        build_hooks([{"type": "wandb", "project": "walk"}], tmp_path)


def test_unknown_hook_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="console"):
        build_hooks([{"type": "consol"}], tmp_path)
