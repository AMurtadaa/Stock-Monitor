"""Synthetic logic checks, not evidence of sentiment-model performance."""

import math

import pytest
from class_project.sentence_analysis import (
    aggregate,
    analyze_document,
    apply_review_weights,
    display_rows,
    split_sentences,
)


class FakeAnalyzer:
    def tokenizer(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split()) + 2))}

    def predict_batch(self, texts):
        output = []
        for text in texts:
            positive, negative = (0.8, 0.1) if "rose" in text else (0.1, 0.8)
            if "unchanged" in text:
                positive, negative = 0.05, 0.05
            probs = {"positive": positive, "negative": negative, "neutral": 1 - positive - negative}
            output.append(
                {
                    "probabilities": probs,
                    "label": max(probs, key=probs.get),
                    "sentiment": positive - negative,
                }
            )
        return output


def test_segmentation_preserves_text_and_common_financial_punctuation():
    text = 'U.S. revenue rose 2.5%. "Demand fell!"\nThe S&P 500 was unchanged.'
    units = split_sentences(text)
    assert units == ["U.S. revenue rose 2.5%.", '"Demand fell!"', "The S&P 500 was unchanged."]
    assert split_sentences("Revenue rose, but demand fell.") == ["Revenue rose, but demand fell."]
    assert split_sentences("A headline without punctuation") == ["A headline without punctuation"]


def test_equal_weight_opposition_is_not_presented_as_uniform_neutrality():
    result = analyze_document(FakeAnalyzer(), "Nasdaq rose. Dow fell.")
    assert result["overall"]["score"] == pytest.approx(0)
    assert result["overall"]["opposing_sentence_scores"]
    assert result["overall"]["net_direction"] == "near-zero"
    assert "Opposing evidence" in result["overall"]["summary"]
    assert sum(row["normalized_weight"] for row in result["sentences"]) == pytest.approx(1)
    assert result["sentences"][0]["contribution"] == pytest.approx(0.35)
    assert len(display_rows(result)) == 2


def test_target_matching_and_weights_are_explicit_not_entity_inference():
    report = analyze_document(
        FakeAnalyzer(),
        "Nasdaq rose. Dow fell.",
        targets="Nasdaq, QQQ",
        mode="Target relevance (experimental)",
    )
    assert report["overall"]["score"] == pytest.approx(0.7 / 3)
    assert report["sentences"][0]["base_weight"] == 2
    assert report["sentences"][1]["base_weight"] == 1
    assert report["sentences"][0]["matched_target_aliases"] == ["Nasdaq"]
    no_substring = analyze_document(
        FakeAnalyzer(), "QQQX rose.", targets="QQQ", mode="Target relevance (experimental)"
    )
    assert no_substring["sentences"][0]["base_weight"] == 1
    with pytest.raises(ValueError, match="target aliases"):
        analyze_document(FakeAnalyzer(), "Nasdaq rose.", mode="Target relevance (experimental)")


def test_duplicate_headline_is_not_counted_twice_and_has_no_automatic_bonus():
    report = analyze_document(FakeAnalyzer(), "Nasdaq rose. Dow fell.", headline="Nasdaq rose.")
    assert report["duplicates_removed"] == 1
    assert len(report["sentences"]) == 2
    assert report["sentences"][0]["source"] == "headline"
    assert report["sentences"][0]["base_weight"] == 1
    assert report["overall"]["score"] == pytest.approx(0)


def test_manual_review_is_auditable_and_does_not_mutate_original():
    report = analyze_document(FakeAnalyzer(), "Nasdaq rose. Dow fell.")
    reviewed = apply_review_weights(report, [[2, 0], [1, 2]], "Review focused on Nasdaq")
    assert reviewed["review_note"] == "Review focused on Nasdaq"
    assert reviewed["overall"]["score"] == pytest.approx(0.7)
    assert not reviewed["overall"]["opposing_sentence_scores"]
    assert report["sentences"][1]["review_multiplier"] == 1
    assert reviewed["sentences"][1]["review_multiplier"] == 0
    for weights in [
        [[1, 0], [2, 0]],
        [[1, -1], [2, 1]],
        [[1, math.nan], [2, 1]],
        [[1, math.inf], [2, 1]],
        [[1, 6], [2, 1]],
        [[1, True], [2, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [3, 1]],
        [[1.5, 1], [2, 1]],
    ]:
        with pytest.raises(ValueError):
            apply_review_weights(report, weights)


def test_empty_input_long_input_truncation_and_neutral_scores():
    with pytest.raises(ValueError, match="Enter"):
        analyze_document(FakeAnalyzer(), " ")
    with pytest.raises(ValueError, match="4,000"):
        analyze_document(FakeAnalyzer(), "x" * 4001)
    long_unit = analyze_document(FakeAnalyzer(), " ".join(["word"] * 130) + ".")
    assert long_unit["sentences"][0]["truncated"]
    assert long_unit["warnings"]
    neutral = analyze_document(FakeAnalyzer(), "Prices were unchanged.")
    assert neutral["overall"]["score"] == 0
    assert not neutral["overall"]["opposing_sentence_scores"]
    with pytest.raises(ValueError, match="one multiplier"):
        aggregate(neutral, [])


def test_real_gradio_components_and_demo_callbacks(monkeypatch):
    import gradio as gr
    from class_project import analyzers, demo

    monkeypatch.setattr(analyzers, "CheckpointAnalyzer", lambda *args: FakeAnalyzer())
    captured = []
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: captured.append(self))
    demo.launch()
    functions = {fn.fn.__name__: fn.fn for fn in captured[0].fns.values() if fn.fn}
    report, table, weights, shown = functions["analyze_passage"](
        "Nasdaq rose. Dow fell.", "", "Nasdaq", "Target relevance (experimental)"
    )
    assert report == shown
    assert len(table) == len(weights) == 2
    revised, rows, shown = functions["review_weights"](report, [[1, 1], [2, 0]])
    assert revised["overall"]["score"] == pytest.approx(0.7)
    assert revised == shown
    with pytest.raises(gr.Error):
        functions["analyze_passage"]("", "", "", "Equal weights")
    with pytest.raises(gr.Error):
        functions["review_weights"](report, [[1, 0], [2, 0]])
