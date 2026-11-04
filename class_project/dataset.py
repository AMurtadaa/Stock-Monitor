"""Explicit Financial PhraseBank import, cleaning, and immutable example splits."""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict

from class_project.common import LABELS, digest, output_path, provenance, read_json, write_json

URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def clean_lines(lines: list[str]):
    grouped = defaultdict(list)
    invalid = 0
    for line in lines:
        text, separator, label = line.rpartition("@")
        text, label = text.strip(), label.strip()
        if not separator or label not in LABELS or not text:
            invalid += 1
            continue
        grouped[normalize(text)].append((text, label))
    examples, conflicts, duplicates = [], 0, 0
    for key, records in sorted(grouped.items()):
        if len({label for _, label in records}) != 1:
            conflicts += len(records)
            continue
        duplicates += len(records) - 1
        text, label = records[0]
        examples.append(
            {"id": digest(key.encode())[:24], "text": text, "label": LABELS.index(label)}
        )
    return examples, {
        "input_rows": len(lines),
        "invalid_rows": invalid,
        "conflicting_rows_excluded": conflicts,
        "duplicates_removed": duplicates,
    }


def prepare():
    from sklearn.model_selection import train_test_split

    destination = output_path("dataset")
    if (destination / "manifest.json").exists():
        print("Reusing frozen dataset:", read_json(destination / "manifest.json")["split_sizes"])
        return
    with urllib.request.urlopen(URL, timeout=90) as response:
        raw = response.read()
    archive = zipfile.ZipFile(io.BytesIO(raw))
    member = next(n for n in archive.namelist() if n.endswith("/Sentences_75Agree.txt"))
    examples, cleaning = clean_lines(archive.read(member).decode("latin-1").splitlines())
    train, rest = train_test_split(
        examples, test_size=0.30, random_state=42, stratify=[e["label"] for e in examples]
    )
    development, test = train_test_split(
        rest, test_size=0.5, random_state=42, stratify=[e["label"] for e in rest]
    )
    splits = {"train": train, "development": development, "test": test}
    for name, rows in splits.items():
        write_json(destination / f"{name}.json", rows)
    manifest = {
        "dataset": "FinancialPhraseBank 75% agreement",
        "url": URL,
        "archive_sha256": digest(raw),
        "cleaning": cleaning,
        "split_seed": 42,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "class_counts": {
            k: dict(Counter(LABELS[e["label"]] for e in v)) for k, v in splits.items()
        },
        "split_ids": {k: [e["id"] for e in v] for k, v in splits.items()},
        "split_sha256": {k: digest((destination / f"{k}.json").read_bytes()) for k in splits},
        "licence": "CC BY-NC-SA 3.0",
        "environment": provenance(),
    }
    write_json(destination / "manifest.json", manifest)
    (destination / "LICENSE.txt").write_bytes(archive.read("FinancialPhraseBank-v1.0/License.txt"))
    print("Dataset prepared:", manifest["split_sizes"], manifest["class_counts"])
    print("Example:", train[0])


def load_split(name: str):
    path = output_path("dataset", f"{name}.json")
    manifest = read_json(output_path("dataset", "manifest.json"))
    if digest(path.read_bytes()) != manifest["split_sha256"][name]:
        raise ValueError("Frozen split changed; restore it before running experiments.")
    return read_json(path)
