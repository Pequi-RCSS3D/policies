import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

import policies.deploy.inference as inference
from policies.cli import main
from policies.core.logs import get_logger
from policies.deploy.registry import ArtifactRegistry, Manifest, artifact_id, write_manifest
from policies.training.control import RUN_CONFIG, ControlMessage, FileControl
from tests.core.fake_distribution import VERSION, installed_acme

Capture = pytest.CaptureFixture[str]
POLICIES = metadata.version("policies")


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """A run directory, marked by the config the trainer writes."""
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / RUN_CONFIG).write_text(
        "trainer:\n"
        "  tunable: [optimizer.lr, steps, render, mode]\n"
        "  bounds: {optimizer.lr: [0.0, 0.01]}\n",
        encoding="utf-8",
    )
    return directory


def test_set_parses_values(run: Path, capsys: Capture) -> None:
    code = main(["set", str(run), "optimizer.lr=1e-4", "steps=64", "render=true", "mode=fast"])
    assert code == 0
    assert "#1" in capsys.readouterr().out
    assert FileControl(run).poll() == [
        ControlMessage(1, {"optimizer.lr": 1e-4, "steps": 64, "render": True, "mode": "fast"})
    ]


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("lr=1", "lr is not tunable. Tunable: mode, optimizer.lr, render, steps"),
        ("optimizer.lr=0.1", "optimizer.lr=0.1 is above the bound 0.01"),
    ],
)
def test_set_checks_the_run_config(run: Path, change: str, error: str, capsys: Capture) -> None:
    assert main(["set", str(run), "steps=1", change]) == 1
    assert capsys.readouterr().err.strip() == f"error: {error}"
    assert FileControl(run).poll() == []


@pytest.mark.parametrize("command", [["set", "optimizer.lr=1e-4"], ["stop"], ["send", "snapshot"]])
def test_control_needs_a_run_directory(tmp_path: Path, command: list[str], capsys: Capture) -> None:
    typo = tmp_path / "rnu"
    assert main([command[0], str(typo), *command[1:]]) == 1
    message = f"error: {typo.resolve()} is not a run directory: it has no {RUN_CONFIG}"
    assert " ".join(capsys.readouterr().err.split()) == message
    assert not typo.exists()


def test_logs_go_to_stderr(run: Path, capsys: Capture) -> None:
    assert main(["set", str(run), "steps=1"]) == 0
    capsys.readouterr()
    get_logger("policies.test").info("hello", step=1)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hello" in captured.err
    assert "step=1" in captured.err


def test_set_rejects_bad_pairs(run: Path, capsys: Capture) -> None:
    assert main(["set", str(run), "nokey"]) == 1
    assert "key=value" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["pause", "resume", "stop", "checkpoint"])
def test_commands(run: Path, command: str, capsys: Capture) -> None:
    assert main([command, str(run)]) == 0
    assert command in capsys.readouterr().out
    assert FileControl(run).poll() == [ControlMessage(1, command=command)]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    registry = ArtifactRegistry(tmp_path / "registry")
    for index, reward in enumerate((1.0, 2.0)):
        fingerprint = str(index) * 64
        directory = tmp_path / "src" / fingerprint
        directory.mkdir(parents=True)
        (directory / "policy.onnx").write_bytes(b"x")
        manifest = Manifest(
            id=artifact_id("walker", fingerprint),
            name="walker",
            fingerprint=fingerprint,
            created=f"2026-01-0{index + 1}T00:00:00",
            spec={"obs": {"size": 10 + index}},
            metrics={"reward": reward},
            parents=("teacher", "coach")[: index * 2],
            files=("policy.onnx",),
        )
        write_manifest(directory, manifest)
        registry.register(directory)
    return tmp_path / "registry"


A = artifact_id("walker", "0" * 64)
B = artifact_id("walker", "1" * 64)


def _rows(capsys: Capture) -> list[list[str]]:
    """Piped table output split into cells."""
    return [line.split() for line in capsys.readouterr().out.splitlines()]


def test_list_show_diff_tag(root: Path, capsys: Capture) -> None:
    assert main(["list", "--root", str(root)]) == 0
    assert _rows(capsys) == [
        ["id", "created", "tags", "reward"],
        [A, "2026-01-01T00:00:00", "-", "1"],
        [B, "2026-01-02T00:00:00", "-", "2"],
    ]

    assert main(["tag", B, "prod", "--root", str(root)]) == 0
    assert capsys.readouterr().out.strip() == f"prod -> {B} (walker)"
    assert main(["list", "--root", str(root), "--tag", "prod"]) == 0
    assert _rows(capsys)[1:] == [[B, "2026-01-02T00:00:00", "prod", "2"]]

    assert main(["show", B, "--root", str(root)]) == 0
    fields = {row[0]: " ".join(row[1:]) for row in _rows(capsys)}
    assert fields["id"] == B
    assert fields["fingerprint"] == "1" * 64
    assert fields["tags"] == "prod"
    assert fields["parents"] == "teacher, coach"
    assert fields["metrics"] == "reward=2"
    assert main(["show", A, "--root", str(root)]) == 0
    assert {row[0]: " ".join(row[1:]) for row in _rows(capsys)}["parents"] == "-"

    assert main(["diff", A, B, "--root", str(root)]) == 0
    assert _rows(capsys) == [
        ["path", A, B],
        ["metrics.reward", "1.0", "2.0"],
        ["parents", "[]", '["teacher",', '"coach"]'],
        ["spec.obs.size", "10", "11"],
    ]


def test_show_requirements(tmp_path: Path, capsys: Capture) -> None:
    directory = tmp_path / "artifact"
    directory.mkdir()
    fingerprint = "3" * 64
    manifest = Manifest(
        id=artifact_id("kicker", fingerprint),
        name="kicker",
        fingerprint=fingerprint,
        requirements={"rcss3d": "0.1.0", "policies": "0.1.0"},
    )
    write_manifest(directory, manifest)
    ArtifactRegistry(tmp_path / "registry").register(directory)
    assert main(["show", manifest.id, "--root", str(tmp_path / "registry")]) == 0
    fields = {row[0]: " ".join(row[1:]) for row in _rows(capsys)}
    assert fields["requirements"] == "policies 0.1.0, rcss3d 0.1.0"


def test_components(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Capture) -> None:
    with installed_acme(tmp_path, monkeypatch):
        assert main(["components", "transforms"]) == 0
    rows = {row[1]: row for row in _rows(capsys)}
    assert rows["name"] == ["registry", "name", "origin", "fields"]
    assert rows["scale"] == ["transforms", "scale", "policies", POLICIES, "factor:", "float"]
    assert rows["acme.mirror"] == [
        "transforms",
        "acme.mirror",
        "acme",
        VERSION,
        "axis:",
        "int",
        "=",
        "0",
    ]


def test_all_components(capsys: Capture) -> None:
    assert main(["components"]) == 0
    listed = {row[0] for row in _rows(capsys)}
    assert {"transforms", "kinds", "trackers", "distributions", "blocks", "storages"} <= listed


def test_unknown_registry(capsys: Capture) -> None:
    assert main(["components", "transform"]) == 1
    assert "unknown registry 'transform'. Did you mean 'transforms'?" in capsys.readouterr().err


def test_components_need_no_torch() -> None:
    code = (
        "import sys\n"
        "from policies.cli import main\n"
        "assert main(['components', 'transforms']) == 0\n"
        "assert 'torch' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_list_marks_missing_metrics(root: Path, tmp_path: Path, capsys: Capture) -> None:
    directory = tmp_path / "runner"
    directory.mkdir()
    (directory / "policy.onnx").write_bytes(b"x")
    fingerprint = "2" * 64
    manifest = Manifest(
        id=artifact_id("runner", fingerprint),
        name="runner",
        fingerprint=fingerprint,
        created="2026-01-03T00:00:00",
        metrics={"speed": 1.23456},
        files=("policy.onnx",),
    )
    write_manifest(directory, manifest)
    ArtifactRegistry(root).register(directory)
    capsys.readouterr()
    assert main(["list", "--root", str(root)]) == 0
    rows = _rows(capsys)
    assert rows[0] == ["id", "created", "tags", "reward", "speed"]
    assert rows[1][3:] == ["1", "-"]
    assert rows[3][3:] == ["-", "1.235"]


def test_errors_print_brackets_verbatim(run: Path, capsys: Capture) -> None:
    assert main(["set", str(run), "=[red]x"]) == 1
    assert capsys.readouterr().err.strip() == "error: expected key=value, got '=[red]x'"


def test_root_from_environment(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: Capture
) -> None:
    monkeypatch.setenv("POLICIES_REGISTRY", str(root))
    assert main(["list", "--name", "walker"]) == 0
    assert A in capsys.readouterr().out


def test_missing_root(monkeypatch: pytest.MonkeyPatch, capsys: Capture) -> None:
    monkeypatch.delenv("POLICIES_REGISTRY", raising=False)
    assert main(["list"]) == 1
    assert "POLICIES_REGISTRY" in capsys.readouterr().err


def test_unknown_artifact(root: Path, capsys: Capture) -> None:
    assert main(["show", "nope", "--root", str(root)]) == 1
    assert "nope" in capsys.readouterr().err


def test_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Capture) -> None:
    results = iter(
        [
            inference.Validation(),
            inference.Validation(inference.ParitySkip.NO_WEIGHTS),
            inference.Validation(inference.ParitySkip.NO_TORCH),
        ]
    )
    seen: list[Path] = []

    def ok(path: Path) -> inference.Validation:
        seen.append(path)
        return next(results)

    def bad(path: Path) -> None:
        raise ValueError("shape mismatch")

    monkeypatch.setattr(inference, "validate_artifact", ok)
    expected = [
        "OK, ONNX Runtime matches PyTorch",
        "OK, parity with PyTorch skipped: no weights.pt",
        "OK, parity with PyTorch skipped: torch is not installed",
    ]
    for line in expected:
        assert main(["validate", str(tmp_path)]) == 0
        assert capsys.readouterr().out.strip() == line
    assert seen == [tmp_path] * 3

    monkeypatch.setattr(inference, "validate_artifact", bad)
    assert main(["validate", str(tmp_path)]) == 1
    assert "shape mismatch" in capsys.readouterr().err


def test_send_custom_command(run: Path, capsys: Capture) -> None:
    assert main(["send", str(run), "snapshot"]) == 0
    assert "snapshot" in capsys.readouterr().out
    assert FileControl(run).poll() == [ControlMessage(1, command="snapshot")]


def test_fetch_by_id_or_tag(root: Path, tmp_path: Path, capsys: Capture) -> None:
    destination = tmp_path / "cache"
    assert main(["fetch", A, str(destination), "--root", str(root)]) == 0
    assert capsys.readouterr().out.strip() == str(destination / A)
    assert main(["tag", B, "prod", "--root", str(root)]) == 0
    capsys.readouterr()
    assert main(["fetch", "walker:prod", str(destination), "--root", str(root)]) == 0
    assert capsys.readouterr().out.strip() == str(destination / B)
    assert (destination / B / "policy.onnx").read_bytes() == b"x"
    assert main(["fetch", "walker:staging", str(destination), "--root", str(root)]) == 1
    assert "no 'walker' artifact tagged 'staging'" in capsys.readouterr().err


def test_untag(root: Path, capsys: Capture) -> None:
    assert main(["tag", B, "prod", "--root", str(root)]) == 0
    capsys.readouterr()
    assert main(["untag", "walker", "prod", "--root", str(root)]) == 0
    assert capsys.readouterr().out.strip() == f"removed prod from {B} (walker)"
    assert ArtifactRegistry(root).tags() == {}
    assert main(["untag", "walker", "prod", "--root", str(root)]) == 1
    assert "no 'walker' artifact tagged 'prod'" in capsys.readouterr().err


def test_piped_tables_end_at_their_content(root: Path, capsys: Capture) -> None:
    assert main(["show", A, "--root", str(root)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines and all(line == line.rstrip() for line in lines)
    assert max(map(len, lines)) < 80
