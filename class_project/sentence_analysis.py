"""Research-only sentence decomposition and transparent, unvalidated weighting.

No new model, financial feature, benchmark score, or training label is introduced.
"""

from __future__ import annotations

import copy
import math
import re

SCORE_THRESHOLD = 0.15
MAX_INPUT_CHARACTERS = 4000
TOKEN_LIMIT = 128


def split_sentences(text):
    """Small offline heuristic, not a linguistic parser. Users inspect the segments."""
    units, start = [], 0
    for match in re.finditer(r"""[.!?]+["”’')\]]*(?:\s+|$)|\n+""", text):
        end = match.end()
        prefix = text[start:end].strip()
        # Avoid common abbreviations and dotted initials, e.g. U.S. and J.
        if re.search(
            r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Inc|Corp|Ltd|vs|e\.g|i\.e)\.|"
            r"\b(?:[A-Za-z]\.)+)\s*$",
            prefix,
            re.IGNORECASE,
        ):
            continue
        if prefix:
            units.append(prefix)
        start = end
    if text[start:].strip():
        units.append(text[start:].strip())
    return units


def _direction(score):
    if score > SCORE_THRESHOLD:
        return "positive"
    if score < -SCORE_THRESHOLD:
        return "negative"
    return "near-zero"


def aggregate(report, multipliers=None):
    """Weighted mean of sentence scores, never labeled a calibrated probability."""
    result = copy.deepcopy(report)
    rows = result["sentences"]
    if multipliers is None:
        multipliers = [1.0] * len(rows)
    if len(multipliers) != len(rows):
        raise ValueError("Supply exactly one multiplier per sentence.")
    for row, multiplier in zip(rows, multipliers, strict=True):
        if isinstance(multiplier, bool):
            raise ValueError("Weights must be numbers between 0 and 5.")
        multiplier = float(multiplier)
        if not math.isfinite(multiplier) or not 0 <= multiplier <= 5:
            raise ValueError("Weights must be finite numbers between 0 and 5.")
        row["review_multiplier"] = multiplier
        row["effective_weight"] = row["base_weight"] * multiplier
    total = sum(row["effective_weight"] for row in rows)
    if total <= 0:
        raise ValueError("At least one sentence must have a nonzero weight.")
    for row in rows:
        row["normalized_weight"] = row["effective_weight"] / total
        row["contribution"] = row["normalized_weight"] * row["sentiment"]
    score = sum(row["contribution"] for row in rows)
    positive_share = sum(
        row["normalized_weight"] for row in rows if row["sentiment"] > SCORE_THRESHOLD
    )
    negative_share = sum(
        row["normalized_weight"] for row in rows if row["sentiment"] < -SCORE_THRESHOLD
    )
    opposing = positive_share > 0 and negative_share > 0
    result["overall"] = {
        "score": score,
        "net_direction": _direction(score),
        "opposing_sentence_scores": opposing,
        "summary": f"Opposing evidence; net {_direction(score)}"
        if opposing
        else f"Net {_direction(score)}",
        "positive_sentence_weight_share": positive_share,
        "negative_sentence_weight_share": negative_share,
        "formula": "sum(normalized_weight * (P(positive) - P(negative)))",
        "not_a_probability": True,
    }
    return result


def analyze_document(analyzer, body, headline="", targets="", mode="Equal weights"):
    body, headline = body.strip(), headline.strip()
    if not body and not headline:
        raise ValueError("Enter a headline or financial passage.")
    if len(body) + len(headline) > MAX_INPUT_CHARACTERS:
        raise ValueError("Please use at most 4,000 characters across headline and passage.")
    if mode not in ["Equal weights", "Target relevance (experimental)"]:
        raise ValueError("Unknown weighting mode.")
    aliases = list(dict.fromkeys(t.strip() for t in targets.split(",") if t.strip()))
    if len(aliases) > 20 or len(targets) > 300:
        raise ValueError("Use at most 20 target aliases and 300 characters.")
    if mode != "Equal weights" and not aliases:
        raise ValueError("Enter target aliases before using target-relevance weights.")
    units = [("headline", headline)] if headline else []
    units += [("sentence", text) for text in split_sentences(body)]
    seen, unique = set(), []
    for source, text in units:
        key = " ".join(text.casefold().split())
        if key not in seen:
            unique.append((source, text))
            seen.add(key)
    if len(unique) > 80:
        raise ValueError("Please shorten the passage to at most 80 sentence units.")
    predictions = analyzer.predict_batch([text for source, text in unique])
    rows = []
    for index, ((source, text), prediction) in enumerate(zip(unique, predictions, strict=True), 1):
        matched = [
            alias
            for alias in aliases
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.IGNORECASE)
        ]
        targeted = mode != "Equal weights" and bool(matched)
        token_count = len(analyzer.tokenizer(text, truncation=False)["input_ids"])
        probabilities = sorted(prediction["probabilities"].values(), reverse=True)
        rows.append(
            {
                "id": index,
                "source": source,
                "text": text,
                **prediction,
                "token_count": token_count,
                "truncated": token_count > TOKEN_LIMIT,
                "class_probability_margin": probabilities[0] - probabilities[1],
                "matched_target_aliases": matched,
                "base_weight": 2.0 if targeted else 1.0,
                "weight_reason": "Explicit target alias match (2x heuristic)"
                if targeted
                else "Equal baseline/context weight (1x)",
            }
        )
    return aggregate(
        {
            "method": "Sentence-level DistilBERT with transparent heuristic aggregation v1",
            "mode": mode,
            "target_aliases": aliases,
            "duplicates_removed": len(units) - len(unique),
            "warnings": [
                "Some units exceed 128 tokens. Their scores do not cover the entire unit; "
                "review or shorten these units before relying on the overall score."
            ]
            if any(row["truncated"] for row in rows)
            else [],
            "sentences": rows,
            "notes": [
                "Experimental demo extension, not evaluated on the original benchmark.",
                "Target matching uses literal aliases, not entity or event understanding.",
                "Headline and position do not automatically make a statement more important.",
                "A high class probability is not an importance or reliability weight.",
                "Opposing scores indicate disagreement across units, not a trained mixed class.",
                "Near-zero can reflect neutrality, uncertainty, or positive/negative cancellation.",
                "Sentence splitting is heuristic; review the displayed segments.",
                "Each unit is truncated to 128 tokens; inspect the truncation flags.",
                "Credibility, severity, timing, factual accuracy and returns are not inferred.",
            ],
        }
    )


def apply_review_weights(report, table, reason=""):
    if not report:
        raise ValueError("Analyze the passage first.")
    if table is None or len(table) != len(report["sentences"]):
        raise ValueError("Keep one review row for each sentence ID.")
    by_id = {}
    for row in table:
        if len(row) != 2:
            raise ValueError("Review rows must contain sentence ID and multiplier only.")
        identifier, multiplier = row
        if isinstance(identifier, bool) or not float(identifier).is_integer():
            raise ValueError("Sentence IDs must be whole numbers.")
        identifier = int(identifier)
        if identifier in by_id:
            raise ValueError("Duplicate sentence ID in review weights.")
        by_id[identifier] = multiplier
    expected = [row["id"] for row in report["sentences"]]
    if set(by_id) != set(expected):
        raise ValueError("Do not add, remove, or change sentence IDs.")
    result = aggregate(report, [by_id[identifier] for identifier in expected])
    result["review_note"] = str(reason).strip()[:1000]
    return result


def display_rows(report):
    return [
        [
            row["id"],
            row["source"],
            row["text"],
            row["label"],
            round(row["sentiment"], 4),
            row["base_weight"],
            row["review_multiplier"],
            round(row["normalized_weight"], 4),
            round(row["contribution"], 4),
            row["truncated"],
            row["weight_reason"],
        ]
        for row in report["sentences"]
    ]
