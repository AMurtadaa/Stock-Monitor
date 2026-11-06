"""Conservative single-process Trainer checkpoint recovery; never delete saved work."""

import json
import time
from pathlib import Path


def checkpoint_problem(path):
    """Check the files required by this project's non-distributed Trainer setup."""
    required = [
        "trainer_state.json",
        "config.json",
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ]
    for name in required:
        file = path / name
        if not file.is_file() or file.stat().st_size == 0:
            return f"missing or empty {name}"
    try:
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        json.loads((path / "config.json").read_text(encoding="utf-8"))
        if state["global_step"] != int(path.name.split("-")[-1]):
            return "trainer state step does not match directory"
        best = state.get("best_model_checkpoint")
        if best and not (Path(best) / "model.safetensors").is_file():
            return "best-model checkpoint referenced by trainer state is unavailable"
    except (OSError, ValueError, KeyError, TypeError):
        return "unreadable or invalid trainer/config JSON"
    return None


def prepare_checkpoint_run(root):
    """Select newest structurally complete save; write into a fresh attempt directory.

    Existing directories, including interrupted saves, remain untouched. These
    checks detect incomplete saves, not arbitrary binary corruption.
    """
    root = Path(root)
    candidates = list(root.glob("checkpoint-*"))
    for attempt in root.glob("attempt-*"):
        if attempt.is_dir():
            candidates.extend(attempt.glob("checkpoint-*"))
    candidates = [p for p in candidates if p.is_dir() and p.name[11:].isdigit()]
    candidates.sort(key=lambda p: (int(p.name[11:]), p.stat().st_mtime_ns), reverse=True)
    previous = None
    for path in candidates:
        problem = checkpoint_problem(path)
        if problem:
            print(f"Skipping incomplete checkpoint {path}: {problem}", flush=True)
        elif previous is None:
            previous = str(path)
    destination = root / f"attempt-{time.time_ns()}" if candidates else root
    if previous:
        print(f"Resuming verified checkpoint: {previous}", flush=True)
    else:
        print(
            "No complete checkpoint to resume; starting this seed from the base model.", flush=True
        )
    print(f"Checkpoint output: {destination}. Existing saves are preserved.", flush=True)
    return destination, previous
