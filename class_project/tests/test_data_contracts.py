"""Dataset split integrity and archive ingestion contract checks."""

import io
import zipfile

import pandas as pd
from class_project import common, dataset
from class_project.common import digest, output_path, read_json
from class_project.financial import filter_news


def test_archive_split_freeze_and_disjointness(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ARTIFACTS", tmp_path)
    lines = [
        f"Sentence {i} is about company {label}@{label}"
        for label in common.LABELS
        for i in range(40)
    ]
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as z:
        z.writestr("FinancialPhraseBank-v1.0/Sentences_75Agree.txt", "\n".join(lines))
        z.writestr("FinancialPhraseBank-v1.0/License.txt", "test fixture")
    monkeypatch.setattr(
        dataset.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(raw.getvalue())
    )
    dataset.prepare()
    splits = [dataset.load_split(name) for name in ["train", "development", "test"]]
    ids = [{r["id"] for r in s} for s in splits]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert sum(len(s) for s in splits) == 120
    before = output_path("dataset", "manifest.json").read_bytes()
    dataset.prepare()
    assert output_path("dataset", "manifest.json").read_bytes() == before


def test_news_dedup_daily_cap_delay_and_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ARTIFACTS", tmp_path / "artifacts")
    rows = [
        {
            "Date": f"2020-01-01 12:{i:02}:00 UTC",
            "Article_title": f"AAPL news {i}",
            "Stock_symbol": "AAPL",
        }
        for i in range(25)
    ]
    rows += [
        rows[0],
        {"Date": "2020-01-01 12:00 UTC", "Article_title": "Outside panel", "Stock_symbol": "ZZZ"},
        {"Date": None, "Article_title": "No date", "Stock_symbol": "AAPL"},
    ]
    csv = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    filter_news(str(csv))
    filtered = pd.read_csv(output_path("financial", "headlines.csv"))
    assert len(filtered) == 20
    delays = pd.to_datetime(filtered.available_at, utc=True) - pd.to_datetime(
        filtered.published, utc=True
    )
    assert (delays == pd.Timedelta(hours=48)).all()
    assert read_json(output_path("financial", "news_manifest.json"))["filtered_sha256"] == digest(
        output_path("financial", "headlines.csv").read_bytes()
    )
