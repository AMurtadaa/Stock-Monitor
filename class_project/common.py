"""Paths, reproducibility, and safe writes for research-only artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "class_artifacts"
LABELS = ["negative", "neutral", "positive"]
BASE_MODEL = "distilbert/distilbert-base-uncased"
PANEL = ["AAPL", "MSFT", "INTC", "CSCO", "JPM", "BAC", "XOM", "CVX", "JNJ", "PFE", "CAT", "BA"]
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Colab exports a notebook-only backend to child processes. Research commands
# save plots to files and must not inherit that kernel-specific backend.
os.environ["MPLBACKEND"] = "Agg"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def output_path(*parts: str) -> Path:
    path = ARTIFACTS.joinpath(*parts).resolve()
    if not path.is_relative_to(ARTIFACTS.resolve()):
        raise ValueError("Research output must stay under class_artifacts.")
    return path


def write_json(path: Path, value) -> None:
    if not path.resolve().is_relative_to(ARTIFACTS.resolve()):
        raise ValueError("Refusing to write outside class_artifacts.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def provenance() -> dict:
    versions = {}
    for package in [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "vaderSentiment",
        "gradio",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {
        "python": platform.python_version(),
        "versions": versions,
        "source_commit": commit.stdout.strip() if commit.returncode == 0 else "archive",
        "upstream_commit": "e26b5c305640a3d1be0b895dfc6720e367507ab3",
    }
