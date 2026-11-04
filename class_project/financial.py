"""Frozen historical-news features and a matched LightGBM ablation, never live models."""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import asdict
from pathlib import Path

from class_project.common import PANEL, digest, output_path, provenance, read_json, write_json


def filter_news(csv_path: str):
    import contextlib
    import io
    import urllib.request

    import pandas as pd

    destination = output_path("financial", "headlines.csv")
    if destination.exists():
        raise ValueError("Filtered news already exists. Reuse it; do not replace frozen inputs.")
    required = ["Date", "Article_title", "Stock_symbol"]
    selected, total = [], 0
    with contextlib.ExitStack() as stack:
        if csv_path.startswith("https://"):
            # Passing a URL directly to pandas can buffer the entire 23 GB file.
            # Hand pandas an open network stream instead, keeping reads bounded.
            response = stack.enter_context(urllib.request.urlopen(csv_path, timeout=120))
            stream = stack.enter_context(
                io.TextIOWrapper(response, encoding="utf-8", errors="replace", newline="")
            )
        else:
            stream = stack.enter_context(
                open(csv_path, encoding="utf-8", errors="replace", newline="")
            )
        reader = pd.read_csv(stream, usecols=required, chunksize=50000)
        for chunk in reader:
            total += len(chunk)
            chunk = chunk[chunk.Stock_symbol.isin(PANEL)].copy()
            chunk["published"] = pd.to_datetime(chunk.Date, utc=True, errors="coerce")
            chunk = chunk.dropna(subset=["published", "Article_title"])
            chunk = chunk[chunk.published.between("2014-12-01", "2023-12-31 23:59:59+00:00")]
            selected.append(chunk[["Stock_symbol", "published", "Article_title"]])
            print(
                f"Scanned {total:,} archive rows; kept {sum(len(s) for s in selected):,}",
                flush=True,
            )
    if not selected or not sum(len(s) for s in selected):
        raise ValueError("No matching historical headlines; inspect source and coverage.")
    news = pd.concat(selected).rename(
        columns={"Stock_symbol": "ticker", "Article_title": "headline"}
    )
    news["normalized"] = (
        news.headline.astype(str).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    news = news[news.normalized.ne("")].sort_values(["published", "normalized"])
    news = news.drop_duplicates(["ticker", "normalized"], keep="first")
    news["day"] = news.published.dt.strftime("%Y-%m-%d")
    news = news.groupby(["ticker", "day"], sort=False).head(20)
    news["available_at"] = news.published + pd.Timedelta(hours=48)
    destination.parent.mkdir(parents=True, exist_ok=True)
    news[["ticker", "published", "available_at", "headline"]].to_csv(destination, index=False)
    write_json(
        output_path("financial", "news_manifest.json"),
        {
            "input_filename": Path(csv_path).name,
            "scanned_rows": total,
            "filtered_rows": len(news),
            "filtered_sha256": digest(destination.read_bytes()),
            "panel": PANEL,
            "availability_delay_hours": 48,
            "first": news.published.min(),
            "last": news.published.max(),
            "by_ticker": news.groupby("ticker").size().to_dict(),
            "environment": provenance(),
            "timestamp_note": (
                "Archive publication times not independently verified; "
                "48h delay is conservative, not proof of exact availability."
            ),
        },
    )
    print("Frozen headlines saved:", destination)


def score_news():
    import sqlite3

    import pandas as pd

    from class_project.analyzers import CheckpointAnalyzer

    run = output_path("nlp", "seed-42")
    manifest = read_json(run / "manifest.json")
    if manifest["smoke"]:
        raise ValueError("Financial experiment requires the full seed-42 model.")
    source = output_path("financial", "headlines.csv")
    source_manifest = read_json(output_path("financial", "news_manifest.json"))
    if digest(source.read_bytes()) != source_manifest["filtered_sha256"]:
        raise ValueError("Filtered headlines changed.")
    if digest((run / "model/model.safetensors").read_bytes()) != manifest["checkpoint_sha256"]:
        raise ValueError("Seed-42 checkpoint changed.")
    news = pd.read_csv(source)
    analyzer = CheckpointAnalyzer(str(run / "model"))
    cache = sqlite3.connect(output_path("financial", "headline_scores.sqlite"))
    cache.execute(
        "CREATE TABLE IF NOT EXISTS scores "
        "(model TEXT, headline TEXT, score REAL, PRIMARY KEY(model,headline))"
    )
    fingerprint = manifest["checkpoint_sha256"]
    scores = dict(cache.execute("SELECT headline,score FROM scores WHERE model=?", (fingerprint,)))
    todo = [h for h in news.headline.unique() if h not in scores]
    start = time.perf_counter()
    try:
        for i in range(0, len(todo), 32):
            batch = todo[i : i + 32]
            values = analyzer.score_batch(batch)
            cache.executemany(
                "INSERT OR REPLACE INTO scores VALUES (?,?,?)",
                [(fingerprint, h, s) for h, s in zip(batch, values, strict=True)],
            )
            cache.commit()
            scores.update(zip(batch, values, strict=True))
            done = min(i + 32, len(todo))
            rate = done / max(time.perf_counter() - start, 0.001)
            print(
                f"Scored {done:,}/{len(todo):,}; {rate:.1f}/s; "
                f"remaining {(len(todo) - done) / max(rate, 0.001) / 60:.1f} min",
                flush=True,
            )
    finally:
        cache.close()
    news["sentiment"] = news.headline.map(scores)
    destination = output_path("financial", "scored_news.csv")
    news.to_csv(destination, index=False)
    write_json(
        output_path("financial", "scoring_manifest.json"),
        {
            "model_sha256": fingerprint,
            "news_sha256": source_manifest["filtered_sha256"],
            "scored_sha256": digest(destination.read_bytes()),
            "rows": len(news),
            "environment": provenance(),
        },
    )


def snapshot(sec_user_agent: str):
    """Freeze prices and filing-dated fundamentals without writing the app database."""
    import os

    from stock_monitor.config import get_settings
    from stock_monitor.providers.edgar_provider import EdgarProvider
    from stock_monitor.providers.yfinance_provider import YFinanceProvider

    if "@" not in sec_user_agent or "example.com" in sec_user_agent:
        raise ValueError(
            "Supply a real SEC contact User-Agent, e.g. ClassResearch/1.0 (your email)."
        )
    os.environ["SEC_USER_AGENT"] = sec_user_agent
    get_settings.cache_clear()
    prices, fundamentals = YFinanceProvider(), EdgarProvider()
    directory = output_path("financial", "snapshot")
    directory.mkdir(parents=True, exist_ok=True)
    for ticker in [*PANEL, "SPY"]:
        price_path = directory / f"{ticker}_prices.csv"
        fact_path = directory / f"{ticker}_facts.json"
        if not price_path.exists():
            frame = prices.get_prices(ticker, dt.date(2013, 1, 1), dt.date(2025, 2, 1))
            if frame.empty:
                raise ValueError(f"No prices for {ticker}; retry later, do not fabricate data.")
            frame.sort_index().to_csv(price_path, index_label="date")
        if ticker != "SPY" and not fact_path.exists():
            facts = fundamentals.get_fundamentals(ticker)
            if not facts:
                raise ValueError(f"No point-in-time fundamentals for {ticker}.")
            write_json(fact_path, [asdict(f) for f in facts])
        print("Snapshot ready:", ticker, flush=True)
        time.sleep(0.25)
    write_json(
        directory / "manifest.json",
        {
            "sha256": {
                p.name: digest(p.read_bytes())
                for p in sorted(directory.iterdir())
                if p.name != "manifest.json"
            },
            "requested_start": "2013-01-01",
            "requested_end_exclusive": "2025-02-01",
            "environment": provenance(),
        },
    )


def news_lookup(news, as_of):
    """Mean of per-available-day means, matching upstream lookup; conservative midnight cutoff."""
    import pandas as pd

    cutoff = pd.Timestamp(as_of, tz="UTC")
    rows = news[
        (news.available_at <= cutoff) & (news.available_at >= cutoff - pd.Timedelta(days=7))
    ]
    if rows.empty:
        return 0.0, 0
    daily = rows.groupby(rows.available_at.dt.floor("D")).sentiment.mean()
    return float(daily.mean()), len(rows)


def partition(frame, start, end, next_period=None):
    import pandas as pd

    selected = frame[frame.as_of.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if next_period:
        selected = selected[selected.label_end < pd.Timestamp(next_period)]
    return selected


def compare():
    import pandas as pd
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    from stock_monitor.features.builder import FEATURE_COLUMNS, build_training_frame
    from stock_monitor.models.scorer import train_model
    from stock_monitor.providers.base import FundamentalFact

    source = output_path("financial", "scored_news.csv")
    if (
        digest(source.read_bytes())
        != read_json(output_path("financial", "scoring_manifest.json"))["scored_sha256"]
    ):
        raise ValueError("Scored news changed.")
    news = pd.read_csv(source)
    news["available_at"] = pd.to_datetime(news.available_at, utc=True)
    directory = output_path("financial", "snapshot")
    manifest = read_json(directory / "manifest.json")
    for filename, expected in manifest["sha256"].items():
        if digest((directory / filename).read_bytes()) != expected:
            raise ValueError(f"Frozen financial snapshot changed: {filename}")

    def prices(ticker):
        return pd.read_csv(
            directory / f"{ticker}_prices.csv", index_col="date", parse_dates=True
        ).sort_index()

    benchmark = prices("SPY")
    frames = []
    for ticker in PANEL:
        facts = read_json(directory / f"{ticker}_facts.json")
        for f in facts:
            for key in ["fiscal_end", "known_on", "period_start"]:
                if f.get(key):
                    f[key] = dt.date.fromisoformat(f[key])
        facts = [FundamentalFact(**f) for f in facts]
        p = prices(ticker)
        ticker_news = news[news.ticker == ticker]

        def lookup(date, source_news=ticker_news):
            return news_lookup(source_news, date)[0]

        frame = build_training_frame(ticker, p, facts, benchmark, 12, sentiment_lookup=lookup)
        if frame.empty:
            raise ValueError(f"No labelled examples for {ticker}.")
        ends, counts = [], []
        for date in frame.as_of:
            target = pd.Timestamp(date) + pd.DateOffset(months=12)
            a, b = p.loc[target:].index, benchmark.loc[target:].index
            ends.append(max(a[0], b[0]) if len(a) and len(b) else pd.NaT)
            counts.append(news_lookup(ticker_news, date)[1])
        frame["label_end"], frame["news_count"] = ends, counts
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame["as_of"] = pd.to_datetime(frame.as_of)
    frame["label_end"] = pd.to_datetime(frame.label_end)
    frame = frame[frame.as_of.between("2015-01-01", "2023-12-31")].dropna(subset=["label_end"])
    if frame.duplicated(["ticker", "as_of"]).any():
        raise ValueError("Duplicate stock/date observations.")
    if (pd.to_datetime(frame.fundamentals_known_on) > frame.as_of).any():
        raise ValueError("A future filing entered a historical feature.")
    splits = {
        "initial_train": partition(frame, "2015-01-01", "2018-12-31", "2020-01-01"),
        "development": partition(frame, "2020-01-01", "2020-12-31", "2022-01-01"),
        "refit": partition(frame, "2015-01-01", "2020-12-31", "2022-01-01"),
        "test": partition(frame, "2022-01-01", "2023-12-31"),
    }
    coverage = {
        name: {
            "rows": len(s),
            "stocks": s.ticker.nunique(),
            "dates": s.as_of.nunique(),
            "positive_rate": float(s.label.mean()) if len(s) else None,
            "news_coverage": float(s.news_count.gt(0).mean()) if len(s) else 0,
            "nonzero_sentiment": int(s.sentiment.ne(0).sum()),
        }
        for name, s in splits.items()
    }
    write_json(output_path("financial", "coverage.json"), coverage)
    frame.to_csv(output_path("financial", "frozen_examples.csv"), index=False)
    for name, s in splits.items():
        if len(s) < (200 if name in ["initial_train", "refit"] else 50) or s.label.nunique() < 2:
            raise ValueError(
                f"Insufficient examples/classes in {name}. "
                "Read coverage.json; dates were not changed."
            )
        if coverage[name]["news_coverage"] < 0.10 or coverage[name]["nonzero_sentiment"] == 0:
            raise ValueError(
                f"Insufficient sentiment coverage in {name}. "
                "Read coverage.json; no all-zero experiment was run."
            )
        if s.ticker.nunique() != len(PANEL):
            raise ValueError(f"Missing panel stocks in {name}; inspect coverage before proceeding.")
    numeric = tuple(c for c in FEATURE_COLUMNS if c != "sentiment")
    results = {
        "coverage": coverage,
        "target": "12-month return greater than SPY",
        "probabilities": "raw, uncalibrated",
        "limitations": (
            "Fixed surviving-company panel; overlapping annual labels; "
            "descriptive comparison, not a trading backtest."
        ),
        "source_examples_sha256": digest(
            output_path("financial", "frozen_examples.csv").read_bytes()
        ),
        "environment": provenance(),
        "models": {},
    }
    for name, columns in [("without_sentiment", numeric), ("with_sentiment", FEATURE_COLUMNS)]:
        variant = {}
        for stage, train_name, test_name in [
            ("development", "initial_train", "development"),
            ("final", "refit", "test"),
        ]:
            train, test = splits[train_name], splits[test_name]
            if train.label_end.max() >= test.as_of.min():
                raise ValueError("Training outcome overlaps evaluation period.")
            model = train_model(train, feature_columns=columns)
            probabilities = model.predict_proba(test[list(columns)])[:, 1]
            variant[stage] = {
                "brier": float(brier_score_loss(test.label, probabilities)),
                "roc_auc": float(roc_auc_score(test.label, probabilities)),
                "accuracy": float(accuracy_score(test.label, probabilities >= 0.5)),
                "n": len(test),
            }
            predicted = test[["ticker", "as_of", "label", "label_end", "news_count"]].copy()
            predicted["probability"] = probabilities
            predicted.to_csv(
                output_path("financial", f"{name}_{stage}_predictions.csv"), index=False
            )
            if stage == "final":
                import joblib

                joblib.dump(model, output_path("financial", f"{name}_model.joblib"))
        results["models"][name] = {
            "feature_columns": list(columns),
            "metrics": variant,
            "parameters": model.get_params(),
        }
    write_json(output_path("financial", "metrics.json"), results)
    print(json.dumps(results, indent=2, default=str), flush=True)
