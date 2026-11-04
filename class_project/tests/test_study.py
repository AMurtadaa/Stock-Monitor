"""Meaningful offline tests for data leakage, metrics and research isolation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from class_project.common import output_path
from class_project.dataset import clean_lines
from class_project.financial import news_lookup, partition
from class_project.nlp import classification_metrics


def test_conflicting_duplicates_excluded_and_final_separator():
    rows, stats = clean_lines(
        [
            "Contact A@B is unchanged.@neutral",
            "  profit up @positive",
            "PROFIT UP@negative",
            "same@positive",
            "same@positive",
            "bad line",
        ]
    )
    assert len(rows) == 2
    assert any("A@B" in r["text"] for r in rows)
    assert stats["conflicting_rows_excluded"] == 2
    assert stats["duplicates_removed"] == 1
    assert stats["invalid_rows"] == 1


def test_late_news_cannot_change_previous_features():
    news = pd.DataFrame(
        {
            "available_at": pd.to_datetime(["2020-01-02", "2020-01-11"], utc=True),
            "sentiment": [0.4, -0.9],
        }
    )
    assert news_lookup(news, "2020-01-05") == (0.4, 1)
    assert news_lookup(news.iloc[:1], "2020-01-05") == news_lookup(news, "2020-01-05")
    assert news_lookup(news, "2019-12-01") == (0.0, 0)


def test_purge_uses_actual_label_end_not_assumed_month():
    frame = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2020-01-01", "2020-12-01"]),
            "label_end": pd.to_datetime(["2021-01-04", "2022-01-03"]),
        }
    )
    assert len(partition(frame, "2020-01-01", "2020-12-31", "2022-01-01")) == 1


def test_metrics_and_output_isolation():
    m = classification_metrics([0, 1, 2], [0, 1, 2])
    assert m["accuracy"] == m["macro_f1"] == 1.0
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ValueError):
        output_path("..", "models", "latest.joblib")


def test_vader_thresholds_and_batch_order():
    from class_project.analyzers import vader_predictions

    rows = vader_predictions(
        ["Company beats earnings, stock soars", "Firm slashes guidance amid fraud probe"]
    )
    assert [r["label"] for r in rows] == ["positive", "negative"]
