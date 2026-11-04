"""Training and held-out evaluation for the sentiment study."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time

from class_project.common import (
    BASE_MODEL,
    LABELS,
    digest,
    output_path,
    provenance,
    read_json,
    write_json,
)
from class_project.dataset import load_split


def classification_metrics(y, predicted):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(
            f1_score(y, predicted, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "per_class": classification_report(
            y, predicted, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
        "n": len(y),
    }


def train(seed=42, smoke=False):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(seed)
    run = output_path("nlp", "smoke" if smoke else f"seed-{seed}")
    if (run / "manifest.json").exists():
        print("Completed run already exists; reusing", run)
        return
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_rows, dev_rows = load_split("train"), load_split("development")
    if smoke:
        train_rows, dev_rows = train_rows[:24], dev_rows[:12]

    def encoded(rows):
        ds = Dataset.from_list(rows)
        return ds.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=128),
            batched=True,
            remove_columns=["id", "text"],
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=dict(enumerate(LABELS)),
        label2id={name: i for i, name in enumerate(LABELS)},
    )
    args = TrainingArguments(
        output_dir=str(run / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        max_steps=2 if smoke else -1,
        weight_decay=0.01,
        eval_strategy="steps" if smoke else "epoch",
        eval_steps=2 if smoke else None,
        save_strategy="steps" if smoke else "epoch",
        save_steps=2 if smoke else 500,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=1 if smoke else 20,
        report_to="none",
        seed=seed,
        data_seed=seed,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=encoded(train_rows),
        eval_dataset=encoded(dev_rows),
        compute_metrics=lambda p: {
            "macro_f1": classification_metrics(p.label_ids, p.predictions.argmax(-1))["macro_f1"]
        },
    )
    previous = (
        get_last_checkpoint(str(run / "checkpoints")) if (run / "checkpoints").exists() else None
    )
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=previous)
    trainer.save_model(str(run / "model"))
    tokenizer.save_pretrained(run / "model")
    # Verify a saved checkpoint produces the same logits as the in-memory model.
    sample = tokenizer(dev_rows[0]["text"], truncation=True, max_length=128, return_tensors="pt")
    trainer.model.eval()
    with torch.inference_mode():
        expected = trainer.model(
            **{k: v.to(trainer.model.device) for k, v in sample.items()}
        ).logits.cpu()
        restored = AutoModelForSequenceClassification.from_pretrained(run / "model").eval()
        actual = restored(**sample).logits
    if not torch.allclose(expected, actual, atol=0.02, rtol=0.02):
        raise RuntimeError("Checkpoint reload changed predictions beyond tolerance.")
    write_json(run / "history.json", trainer.state.log_history)
    write_json(
        run / "manifest.json",
        {
            "seed": seed,
            "smoke": smoke,
            "base_model": BASE_MODEL,
            "base_revision": getattr(model.config, "_commit_hash", None),
            "split_fingerprint": digest(output_path("dataset", "manifest.json").read_bytes()),
            "training_examples": len(train_rows),
            "development_examples": len(dev_rows),
            "arguments": args.to_dict(),
            "metrics": result.metrics,
            "elapsed_seconds": time.perf_counter() - started,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "checkpoint_sha256": digest((run / "model/model.safetensors").read_bytes()),
            "environment": provenance(),
            "reload_check": "passed",
        },
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    (run / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    print("Saved model, history, environment and manifest:", run, flush=True)


def evaluate(smoke=False, finbert=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from class_project.analyzers import CheckpointAnalyzer, vader_predictions

    rows = load_split("development")[:12] if smoke else load_split("test")
    texts = [r["text"] for r in rows]
    truth = [r["label"] for r in rows]
    destination = output_path("evaluation", "smoke" if smoke else "full")
    if (destination / "metrics.json").exists():
        print("Reusing saved evaluation:", destination)
        return
    seeds = [42] if smoke else [42, 17, 73]
    # Check all requested models exist before touching the held-out labels.
    for seed in seeds:
        run = output_path("nlp", "smoke" if smoke else f"seed-{seed}")
        manifest = read_json(run / "manifest.json")
        if manifest["smoke"] != smoke:
            raise ValueError("Smoke and full experiment artifacts cannot be mixed.")
        if manifest["split_fingerprint"] != digest(
            output_path("dataset", "manifest.json").read_bytes()
        ):
            raise ValueError("Model was trained with a different frozen dataset split.")
        if manifest["checkpoint_sha256"] != digest(
            (run / "model/model.safetensors").read_bytes()
        ):
            raise ValueError("Saved checkpoint changed after training.")
    results = {"scope": "development smoke only" if smoke else "held-out test", "models": {}}
    predictions = {"vader": vader_predictions(texts)}
    for seed in seeds:
        key = "smoke" if smoke else f"seed-{seed}"
        analyzer = CheckpointAnalyzer(str(output_path("nlp", key, "model")))
        start = time.perf_counter()
        predictions[key] = analyzer.predict_batch(texts)
        results["models"][key] = {"seconds": time.perf_counter() - start}
        del analyzer
    if finbert:
        predictions["finbert_reference_overlap"] = CheckpointAnalyzer(
            "ProsusAI/finbert"
        ).predict_batch(texts)
    for name, pred in predictions.items():
        metrics = classification_metrics(truth, [LABELS.index(p["label"]) for p in pred])
        results["models"].setdefault(name, {}).update(metrics)
        write_json(
            destination / f"{name}_predictions.json",
            [{**r, "prediction": p} for r, p in zip(rows, pred, strict=True)],
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(metrics["confusion_matrix"], cmap="Blues")
        ax.set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=LABELS,
            yticklabels=LABELS,
            xlabel="Predicted",
            ylabel="Human label",
            title=name,
        )
        for i, values in enumerate(metrics["confusion_matrix"]):
            for j, v in enumerate(values):
                ax.text(j, i, str(v), ha="center", va="center")
        fig.tight_layout()
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / f"{name}_confusion.png", dpi=160)
        plt.close(fig)
    if not smoke:
        results["three_seed_summary"] = {
            m: {
                "mean": statistics.mean(results["models"][f"seed-{s}"][m] for s in seeds),
                "sample_std": statistics.stdev(results["models"][f"seed-{s}"][m] for s in seeds),
            }
            for m in ["accuracy", "macro_f1"]
        }
        p = predictions["seed-42"]
        wrong = [i for i, r in enumerate(rows) if p[i]["label"] != LABELS[r["label"]]]
        right = [i for i, r in enumerate(rows) if p[i]["label"] == LABELS[r["label"]]]
        write_json(
            destination / "review_15_examples.json",
            [
                {**rows[i], "prediction": p[i], "review_note": "Add your explanation here."}
                for i in (wrong[:10] + right[:5])
            ],
        )
        for seed in seeds:
            history = read_json(output_path("nlp", f"seed-{seed}", "history.json"))
            fig, axes = plt.subplots(1, 2, figsize=(9, 3))
            loss = [r for r in history if "loss" in r]
            dev = [r for r in history if "eval_macro_f1" in r]
            axes[0].plot([r["epoch"] for r in loss], [r["loss"] for r in loss])
            axes[0].set_title("Training loss")
            axes[1].plot([r["epoch"] for r in dev], [r["eval_macro_f1"] for r in dev])
            axes[1].set_title("Development macro-F1")
            fig.tight_layout()
            fig.savefig(destination / f"seed-{seed}_curves.png", dpi=160)
            plt.close(fig)
    results["finbert_note"] = (
        "Pretrained FinBERT saw Financial PhraseBank; "
        "reference only, not an unseen-data competitor."
    )
    write_json(destination / "metrics.json", results)
    print(json.dumps(results, indent=2), flush=True)
