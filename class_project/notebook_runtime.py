"""One observable command runner for Colab setup, experiments, and reruns.

Only standard-library imports are allowed here: it must work before pip is installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

WORKFLOW_VERSION = "2.0"


class NotebookSession:
    def __init__(self, repo, python=None, *, heartbeat_seconds=15, emit=print):
        self.repo = Path(repo).resolve()
        self.python = str(python or sys.executable)
        self.heartbeat_seconds = heartbeat_seconds
        self.emit = emit
        self.artifacts = self.repo / "class_artifacts"
        self.logs = self.artifacts / "setup_logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    def _say(self, message):
        self.emit(message, flush=True)

    def command(self, args, *, label):
        """Stream merged output; heartbeat during silence; terminate on interruption."""
        label_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
        log = self.logs / f"{time.time_ns()}_{label_slug}.log"
        self._say(f"START: {label}")
        environment = dict(os.environ)
        environment.update(
            MPLBACKEND="Agg",
            USE_TF="0",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
            PYTHONIOENCODING="utf-8",
            PIP_PROGRESS_BAR="off",
            PIP_DISABLE_PIP_VERSION_CHECK="1",
        )
        began = last_message = time.monotonic()
        process = subprocess.Popen(
            list(map(str, args)),
            cwd=self.repo,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=os.name != "nt",
        )
        messages = queue.Queue()

        def read_output():
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            with log.open("w", encoding="utf-8") as stream:
                while True:
                    try:
                        line = messages.get(timeout=self.heartbeat_seconds)
                    except queue.Empty:
                        self._say(
                            f"Still running: {label} | {time.monotonic() - began:.0f}s elapsed | "
                            f"{time.monotonic() - last_message:.0f}s since last output"
                        )
                        continue
                    if line is None:
                        break
                    last_message = time.monotonic()
                    stream.write(line)
                    stream.flush()
                    self.emit(line, end="", flush=True)
                result = process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            self._say(f"STOPPED: {label}. Saved checkpoints and Drive files were not deleted.")
            raise
        finally:
            reader.join(timeout=2)
            process.stdout.close()
        if result:
            self._say(f"FAILED: {label} (exit {result}). Full output: {log}")
            raise RuntimeError(f"{label} failed. Read the printed error above; log: {log}")
        self._say(f"PASS: {label} ({time.monotonic() - began:.1f}s). Log: {log.name}")

    def setup(self, env_path, *, require_gpu=True):
        env = Path(env_path)
        target = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not target.exists():
            self.command(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "venv",
                    "--system-site-packages",
                    "--without-pip",
                    env,
                ],
                label="Create isolated environment",
            )
        self.python = str(target)
        # A notebook kernel's pip can manage a venv that has no pip/ensurepip.
        self.command(
            [sys.executable, "-u", "-m", "pip", "--python", self.python, "install", "pip"],
            label="Bootstrap pip",
        )
        self.command([self.python, "-u", "-m", "pip", "--version"], label="Verify pip")
        self.command(
            [
                self.python,
                "-u",
                "-m",
                "pip",
                "install",
                "--progress-bar",
                "off",
                "-r",
                "class_project/requirements.txt",
            ],
            label="Install class dependencies",
        )
        self.run("check")
        if require_gpu:
            self.command(
                [
                    self.python,
                    "-u",
                    "-c",
                    "import torch; "
                    "assert torch.cuda.is_available(), 'Select a GPU runtime in Colab'; "
                    "print('GPU READY:', torch.cuda.get_device_name(0))",
                ],
                label="Verify GPU",
            )
        self.command(
            [self.python, "-u", "-m", "pytest", "class_project/tests", "-q"],
            label="Run correctness tests",
        )
        self._say("SETUP PASSED. Continue to dataset preparation.")

    def run(self, *args):
        self.command([self.python, "-u", "-m", "class_project", *args], label=" ".join(args[:2]))

    def smoke(self):
        self._say("Smoke test: two training steps, save/reload, development scoring.")
        self.run("train", "--smoke")
        self.run("evaluate", "--smoke")
        result = self._verify_smoke()
        (self.artifacts / "setup_logs/smoke_passed.json").write_text(
            json.dumps(
                {"workflow": WORKFLOW_VERSION, "reload": "passed", "scope": result["scope"]}
            ),
            encoding="utf-8",
        )
        self._say(
            "SMOKE TEST PASSED: checkpoint verified, reload passed, development metrics exist."
        )
        self._say(
            "These are wiring checks, not final model accuracy. Full training is now allowed."
        )

    def _verify_smoke(self):
        run = self.artifacts / "nlp/smoke"
        metrics = self.artifacts / "evaluation/smoke/metrics.json"
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        result = json.loads(metrics.read_text(encoding="utf-8"))
        if not manifest.get("smoke") or manifest.get("reload_check") != "passed":
            raise RuntimeError("Smoke training did not produce a verified reload manifest.")
        if result.get("scope") != "development smoke only":
            raise RuntimeError("Smoke evaluation has the wrong data scope.")
        if not (run / "model/model.safetensors").exists():
            raise RuntimeError("Smoke checkpoint is missing.")
        for path, expected in [
            (run / "model/model.safetensors", manifest["checkpoint_sha256"]),
            (self.artifacts / "dataset/manifest.json", manifest["split_fingerprint"]),
        ]:
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
            if checksum.hexdigest() != expected:
                raise RuntimeError(f"Smoke artifacts changed: {path}. Do not use stale results.")
        return result

    def full_training(self):
        passed = self.artifacts / "setup_logs/smoke_passed.json"
        if not passed.exists():
            raise RuntimeError("Run section 4 first: it must print SMOKE TEST PASSED.")
        self._verify_smoke()
        for seed in [42, 17, 73]:
            self.run("train", "--seed", str(seed))
        self._say("ALL THREE TRAINING RUNS FINISHED. Continue to held-out evaluation.")
