"""Maintainer utility: build a clean V2 without overwriting the user's saved notebook."""

import json
from pathlib import Path


def build():
    root = Path(__file__).resolve().parent
    original = json.loads((root / "Financial_Sentiment_Study.ipynb").read_text(encoding="utf-8"))
    originals = {c.get("id"): c for c in original["cells"]}
    cells = []

    def cell(kind, text, name):
        item = {
            "cell_type": kind,
            "id": name,
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell(
        "markdown",
        """# Financial Sentiment Study — V2

**Use this clean notebook in Colab. Do not paste in repair cells from the old version.**

[Open V2 in Google Colab](https://colab.research.google.com/github/AMurtadaa/Stock-Monitor/blob/class/financial-sentiment-study/class_project/Financial_Sentiment_Study_V2.ipynb)

GitHub shows the notebook; **Colab runs it**. Select **Runtime → Change runtime type → GPU**.
Run sections 1–4 in order. Section 4 must explicitly print **SMOKE TEST PASSED** before
section 5 allows full training. Every command prints START/PASS/FAILED and a heartbeat
after 15 seconds without new output. Saved Drive data and checkpoints are reused.
This notebook does not modify your deployed application.
""",
        "v2-intro",
    )
    cell(
        "markdown",
        """## 1. Load the current workflow and preserve Drive artifacts

First stop any cell still running in the old notebook. This uses a new code checkout
and a new isolated environment, while keeping your existing Drive dataset/checkpoints.
No restart, deletion or manual package patch is needed. Authorize Drive when prompted.
""",
        "v2-load-info",
    )
    cell(
        "code",
        """import importlib.util
import os
import subprocess
from pathlib import Path

print("FINANCIAL SENTIMENT WORKFLOW V2 — loading", flush=True)
USE_DRIVE = True
repo = Path("/content/stock-monitor-class-v2")
if not repo.exists():
    print("Downloading the current class branch...", flush=True)
    downloaded = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "class/financial-sentiment-study",
         "https://github.com/AMurtadaa/Stock-Monitor.git", str(repo)],
        capture_output=True, text=True,
    )
    print(downloaded.stdout + downloaded.stderr, flush=True)
    downloaded.check_returncode()
else:
    print("Updating the V2 code checkout without resetting local changes...", flush=True)
    updated = subprocess.run(
        ["git", "-C", str(repo), "pull", "--ff-only"],
        capture_output=True, text=True,
    )
    print(updated.stdout + updated.stderr, flush=True)
    updated.check_returncode()
os.chdir(repo)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("SOURCE COMMIT:", commit, flush=True)
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    saved = Path("/content/drive/MyDrive/CSCI390_Financial_Sentiment/class_artifacts")
    saved.mkdir(parents=True, exist_ok=True)
    local = repo / "class_artifacts"
    if not local.exists() and not local.is_symlink():
        local.symlink_to(saved, target_is_directory=True)
    if local.resolve() != saved.resolve():
        raise RuntimeError("Artifacts exist outside Drive. Export before changing storage.")
spec = importlib.util.spec_from_file_location(
    "class_notebook_runtime_v2", repo / "class_project/notebook_runtime.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
assert runtime.WORKFLOW_VERSION == "2.0", "Open the current V2 notebook."
app = runtime.NotebookSession(repo)
print("WORKFLOW:", runtime.WORKFLOW_VERSION, flush=True)
print("ARTIFACTS:", app.artifacts.resolve(), flush=True)
print("SECTION 1 PASSED. Continue to setup.", flush=True)
""",
        "v2-load",
    )
    cell(
        "markdown",
        """## 2. One setup cell, with visible progress

This creates pip if absent, installs dependencies, checks imports/GPU and runs tests.
Installation may take several minutes. A heartbeat means the command is still running,
not that it has passed. Errors are printed in full and saved to `class_artifacts/setup_logs`.
Interrupting a command terminates its subprocess group in Colab instead of leaving
an invisible training/install process behind. Wait for **SETUP PASSED**.
""",
        "v2-setup-info",
    )
    cell("code", 'app.setup("/content/stock-class-env-v2", require_gpu=True)\n', "v2-setup")
    cell(
        "markdown",
        "## 3. Prepare and inspect the human-labeled data\n"
        "Existing frozen splits are reused. Wait for PASS.\n",
        "v2-data-info",
    )
    cell(
        "code",
        """app.run("prepare")
import json

manifest = json.loads((app.artifacts / "dataset/manifest.json").read_text())
print("SPLITS:", manifest["split_sizes"], flush=True)
print("LABEL COUNTS:", manifest["class_counts"], flush=True)
""",
        "v2-data",
    )
    cell(
        "markdown",
        """## 4. Smoke test — require an explicit result

This checks two training steps, saving, reloading and development-only evaluation.
It checks actual output files after the commands finish. Reused successful runs still
produce a visible result. Wait for **SMOKE TEST PASSED**. Low smoke accuracy is expected.
""",
        "v2-smoke-info",
    )
    cell("code", "app.smoke()\n", "v2-smoke")
    cell(
        "markdown",
        "## 5. Three full training runs\nThis is gated on section 4. "
        "Keep the runtime connected; checkpoints go to Drive.\n",
        "v2-full-info",
    )
    cell("code", "app.full_training()\n", "v2-full")
    # Reuse the reviewed experiment descriptions, with one runner for every command.
    pairs = [
        ("938c804e27f84196a10c8828c723f798", "504fb2a444614c0babb325280ed9130a"),
        ("59bbdb311c014d738909a11f9e486628", "b43b363d81ae4b689946ece5c682cd59"),
        ("8a65eabff63a45729fe45fb5ade58bdc", "c3933fab20d04ec698c2621248eb3be0"),
        ("4dd4641cc4064e0191573fe9c69df29b", "8309879909854d7188b41380fd92a7c3"),
        ("3ed186c9a28b402fb0bc4494df01f08d", "cb1e1581032b452c9409d6c6813c49d1"),
    ]
    for description_id, code_id in pairs:
        cell("markdown", "".join(originals[description_id]["source"]), "v2-" + description_id)
        text = "".join(originals[code_id]["source"])
        text = text.replace('run("', 'app.run("')
        cell("code", text, "v2-" + code_id)
    result = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "Financial_Sentiment_Study_V2.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = root / "Financial_Sentiment_Study_V2.ipynb"
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    build()
