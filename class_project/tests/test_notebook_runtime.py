"""Regression checks for the actual Colab command runner, without network or GPU."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from class_project.notebook_runtime import NotebookSession


def session(tmp_path):
    output = []
    app = NotebookSession(
        tmp_path,
        heartbeat_seconds=0.05,
        emit=lambda message, **kwargs: output.append(message),
    )
    return app, output


def test_output_errors_environment_and_heartbeat(tmp_path):
    app, output = session(tmp_path)
    app.command(
        [
            sys.executable,
            "-u",
            "-c",
            "import os, sys, time; print('visible stdout'); "
            "print('visible stderr', file=sys.stderr); "
            "assert os.environ['MPLBACKEND'] == 'Agg'; time.sleep(.2)",
        ],
        label="test stream",
    )
    combined = "\n".join(output)
    assert "visible stdout" in combined
    assert "visible stderr" in combined
    assert "Still running:" in combined
    assert "PASS: test stream" in combined
    assert "visible stdout" in next(app.logs.glob("*.log")).read_text()


def test_nonzero_exit_is_visible_and_stops(tmp_path):
    app, output = session(tmp_path)
    with pytest.raises(RuntimeError, match="failed"):
        app.command([sys.executable, "-c", "raise ValueError('specific failure')"], label="bad")
    assert "specific failure" in "\n".join(output)
    assert "FAILED: bad" in "\n".join(output)


def test_full_training_requires_verified_smoke(tmp_path):
    app, output = session(tmp_path)
    with pytest.raises(RuntimeError, match="section 4"):
        app.full_training()
    calls = []
    app.run = lambda *args: calls.append(args)
    model = app.artifacts / "nlp/smoke/model"
    model.mkdir(parents=True)
    (model / "model.safetensors").touch()
    dataset = app.artifacts / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}")
    (model.parent / "manifest.json").write_text(
        json.dumps(
            {
                "smoke": True,
                "reload_check": "passed",
                "checkpoint_sha256": hashlib.sha256(b"").hexdigest(),
                "split_fingerprint": hashlib.sha256(b"{}").hexdigest(),
            }
        )
    )
    evaluation = app.artifacts / "evaluation/smoke"
    evaluation.mkdir(parents=True)
    (evaluation / "metrics.json").write_text(json.dumps({"scope": "development smoke only"}))
    app.smoke()
    assert "SMOKE TEST PASSED" in "\n".join(output)
    app.full_training()
    assert calls == [
        ("train", "--smoke"),
        ("evaluate", "--smoke"),
        ("train", "--seed", "42"),
        ("train", "--seed", "17"),
        ("train", "--seed", "73"),
    ]
    (model / "model.safetensors").write_bytes(b"changed checkpoint")
    with pytest.raises(RuntimeError, match="artifacts changed"):
        app.full_training()


def test_smoke_cannot_succeed_without_files(tmp_path):
    app, output = session(tmp_path)
    app.run = lambda *args: None
    with pytest.raises(FileNotFoundError):
        app.smoke()
    assert "SMOKE TEST PASSED" not in "\n".join(output)


def test_setup_bootstraps_pip_before_dependency_install(tmp_path):
    app, output = session(tmp_path)
    calls = []
    app.command = lambda args, **kwargs: calls.append((list(map(str, args)), kwargs["label"]))
    app.setup(tmp_path / "fresh-env", require_gpu=True)
    labels = [label for args, label in calls]
    assert labels == [
        "Create isolated environment",
        "Bootstrap pip",
        "Verify pip",
        "Install class dependencies",
        "check",
        "Verify GPU",
        "Run correctness tests",
    ]
    assert "--without-pip" in calls[0][0]
    assert "--python" in calls[1][0]
    assert "SETUP PASSED" in "\n".join(output)


def test_setup_failure_never_claims_success(tmp_path):
    app, output = session(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("installation failed")

    app.command = fail
    with pytest.raises(RuntimeError, match="installation failed"):
        app.setup(tmp_path / "fresh-env")
    assert "SETUP PASSED" not in "\n".join(output)


def test_v2_cells_are_valid_and_have_only_one_runner():
    notebook = Path(__file__).parents[1] / "Financial_Sentiment_Study_V2.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    code = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    for index, source in enumerate(code):
        compile(source, f"notebook_cell_{index}", "exec")
        assert "def run(" not in source
    assert any("app.smoke()" in source for source in code)
    assert any("app.full_training()" in source for source in code)
    assert all(not cell.get("outputs") for cell in cells)
