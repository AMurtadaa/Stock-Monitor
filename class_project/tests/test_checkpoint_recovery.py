"""Interrupted checkpoint regressions, including a real tiny Trainer resume."""

import json

from class_project.checkpoint_recovery import prepare_checkpoint_run


def saved(root, step):
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    for name in ["model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"]:
        (path / name).write_bytes(b"test placeholder, not actual model weights")
    (path / "config.json").write_text("{}")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return path


def test_incomplete_only_restarts_without_touching_files(tmp_path, capsys):
    incomplete = saved(tmp_path, 151)
    (incomplete / "trainer_state.json").unlink()
    before = {p.name: p.read_bytes() for p in incomplete.iterdir()}
    destination, previous = prepare_checkpoint_run(tmp_path)
    assert previous is None
    assert destination.parent == tmp_path and destination.name.startswith("attempt-")
    assert before == {p.name: p.read_bytes() for p in incomplete.iterdir()}
    assert "missing or empty trainer_state.json" in capsys.readouterr().out


def test_older_complete_checkpoint_and_recovery_attempts(tmp_path):
    earlier = saved(tmp_path, 151)
    partial = saved(tmp_path, 302)
    (partial / "optimizer.pt").unlink()
    _, previous = prepare_checkpoint_run(tmp_path)
    assert previous == str(earlier)
    recovered = saved(tmp_path / "attempt-123", 303)
    _, previous = prepare_checkpoint_run(tmp_path)
    assert previous == str(recovered)


def test_invalid_json_and_missing_best_model_are_skipped(tmp_path):
    first = saved(tmp_path, 151)
    (first / "trainer_state.json").write_text("interrupted JSON{")
    second = saved(tmp_path, 302)
    (second / "trainer_state.json").write_text(
        json.dumps({"global_step": 302, "best_model_checkpoint": str(tmp_path / "missing")})
    )
    _, previous = prepare_checkpoint_run(tmp_path)
    assert previous is None


def test_real_tiny_trainer_resume_with_incomplete_newer_save(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_TF", "0")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    import torch
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = DistilBertConfig(
            vocab_size=20, dim=12, hidden_dim=24, n_layers=1, n_heads=2, num_labels=3
        )
        data = [
            {"input_ids": [2, 4, 5, 3], "attention_mask": [1] * 4, "labels": i % 3}
            for i in range(8)
        ]

        def trainer(path, steps):
            return Trainer(
                model=DistilBertForSequenceClassification(config),
                args=TrainingArguments(
                    output_dir=str(path),
                    use_cpu=True,
                    max_steps=steps,
                    per_device_train_batch_size=2,
                    save_strategy="steps",
                    save_steps=1,
                    report_to="none",
                    disable_tqdm=True,
                ),
                train_dataset=data,
            )

        trainer(tmp_path, 2).train()
        broken = tmp_path / "checkpoint-151"
        broken.mkdir()
        (broken / "model.safetensors").write_bytes(b"partial")
        destination, previous = prepare_checkpoint_run(tmp_path)
        assert previous == str(tmp_path / "checkpoint-2")
        resumed = trainer(destination, 3)
        resumed.train(resume_from_checkpoint=previous)
        assert resumed.state.global_step == 3
        assert (broken / "model.safetensors").read_bytes() == b"partial"
    finally:
        torch.set_num_threads(previous_threads)
