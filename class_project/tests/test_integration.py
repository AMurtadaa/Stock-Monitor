"""Synthetic end-to-end checks; these scores are never real research results."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_named_checkpoint_batch_reload_truncation(tmp_path):
    pytest.importorskip("torch")
    from class_project.analyzers import CheckpointAnalyzer
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncompany\nprofit\nfalls\n", encoding="utf-8"
    )
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(tmp_path)
    config = DistilBertConfig(
        vocab_size=8,
        dim=24,
        hidden_dim=32,
        n_layers=1,
        n_heads=2,
        num_labels=3,
        id2label={0: "positive", 1: "negative", 2: "neutral"},
    )
    DistilBertForSequenceClassification(config).save_pretrained(tmp_path)
    reader = CheckpointAnalyzer(str(tmp_path), device="cpu")
    single = reader.predict_batch(["company profit"])[0]
    batch = reader.predict_batch(["company profit", "falls " * 600])
    assert single["probabilities"] == pytest.approx(batch[0]["probabilities"], abs=1e-6)
    assert reader.score("company profit") == pytest.approx(
        single["probabilities"]["positive"] - single["probabilities"]["negative"]
    )
    assert set(batch[1]["probabilities"]) == {"positive", "negative", "neutral"}
    from class_project.sentence_analysis import analyze_document

    passage = analyze_document(reader, "company profit. falls.")
    assert len(passage["sentences"]) == 2
    assert sum(row["contribution"] for row in passage["sentences"]) == pytest.approx(
        passage["overall"]["score"]
    )
    assert reader.score_batch([]) == []
    with pytest.raises(ValueError):
        reader.score(" ")
    with pytest.raises((OSError, ValueError)):
        CheckpointAnalyzer(str(tmp_path / "missing"))


def test_real_feature_builder_and_matched_financial_comparison(tmp_path, monkeypatch):
    from class_project import common
    from class_project.common import PANEL, digest, output_path, read_json, write_json
    from class_project.financial import compare

    monkeypatch.setattr(common, "ARTIFACTS", tmp_path)
    snapshot = output_path("financial", "snapshot")
    snapshot.mkdir(parents=True)
    dates = pd.bdate_range("2013-01-01", "2025-01-31")
    x = np.arange(len(dates))
    for i, ticker in enumerate([*PANEL, "SPY"]):
        values = 100 * np.exp(0.0002 * x + 0.20 * np.sin(x / 170 + i))
        pd.DataFrame(
            {
                "open": values,
                "high": values * 1.01,
                "low": values * 0.99,
                "close": values,
                "volume": 100000,
            },
            index=dates,
        ).to_csv(snapshot / f"{ticker}_prices.csv", index_label="date")
        if ticker != "SPY":
            write_json(
                snapshot / f"{ticker}_facts.json",
                [
                    {
                        "ticker": ticker,
                        "concept": "Assets",
                        "value": 100.0,
                        "unit": "USD",
                        "fiscal_end": "2013-12-31",
                        "known_on": "2014-02-01",
                        "form": "10-K",
                        "period_start": None,
                    }
                ],
            )
    write_json(
        snapshot / "manifest.json",
        {"sha256": {p.name: digest(p.read_bytes()) for p in snapshot.iterdir()}},
    )
    rows = [
        {"ticker": ticker, "available_at": date.isoformat(), "sentiment": 0.4 if i % 2 else -0.4}
        for ticker in PANEL
        for i, date in enumerate(pd.date_range("2014-12-01", "2023-12-31", freq="3D", tz="UTC"))
    ]
    pd.DataFrame(rows).to_csv(output_path("financial", "scored_news.csv"), index=False)
    write_json(
        output_path("financial", "scoring_manifest.json"),
        {"scored_sha256": digest(output_path("financial", "scored_news.csv").read_bytes())},
    )
    compare()
    result = read_json(output_path("financial", "metrics.json"))
    assert "sentiment" not in result["models"]["without_sentiment"]["feature_columns"]
    assert "sentiment" in result["models"]["with_sentiment"]["feature_columns"]
    a = pd.read_csv(output_path("financial", "without_sentiment_final_predictions.csv"))
    b = pd.read_csv(output_path("financial", "with_sentiment_final_predictions.csv"))
    pd.testing.assert_frame_equal(a.drop(columns="probability"), b.drop(columns="probability"))
    assert a.as_of.min() >= "2022-01-01"
    assert len(a) == 24 * len(PANEL)
