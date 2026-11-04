"""Run with: python -m class_project --help. All outputs stay in class_artifacts/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from class_project.common import ARTIFACTS, PANEL, ROOT, output_path, provenance

# Allows a source checkout without installing the whole production application.
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Verify dependencies, GPU and pinned upstream interfaces.")
    sub.add_parser("prepare", help="Download/clean PhraseBank and freeze splits.")
    p = sub.add_parser("train")
    p.add_argument("--seed", type=int, choices=[42, 17, 73], default=42)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finbert", action="store_true")
    p = sub.add_parser("filter-news")
    p.add_argument(
        "--csv", required=True, help="Existing FNSPID archive file, or its public URL (streamed)."
    )
    sub.add_parser("score-news", help="Resumable headline scoring with the full seed-42 model.")
    p = sub.add_parser("snapshot")
    p.add_argument("--sec-user-agent", required=True)
    sub.add_parser("compare", help="Freeze matched examples and run LightGBM ablation.")
    p = sub.add_parser("demo")
    p.add_argument("--share", action="store_true")
    p.add_argument("--finbert", action="store_true")
    p = sub.add_parser("inspect-data")
    p.add_argument(
        "--path", help="Optional existing local archive folder; reports names/sizes only."
    )
    sub.add_parser(
        "export", help="Export study artifacts, excluding external credentials and data stores."
    )
    sub.add_parser(
        "writeup", help="Build an evidence-based report draft and student result summary."
    )
    args = parser.parse_args()
    if args.command == "check":
        import importlib

        import torch

        for dependency in [
            "transformers",
            "datasets",
            "lightgbm",
            "gradio",
            "stock_monitor.models.scorer",
        ]:
            importlib.import_module(dependency)
        from stock_monitor.features.builder import FEATURE_COLUMNS
        from stock_monitor.sentiment import VaderAnalyzer

        if VaderAnalyzer().score("Company beats earnings") <= 0:
            raise RuntimeError("VADER import/score check failed.")
        print(
            json.dumps(
                {
                    **provenance(),
                    "gpu": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "CPU (use GPU in Colab)",
                    "features": FEATURE_COLUMNS,
                    "panel": PANEL,
                },
                indent=2,
            )
        )
    elif args.command == "prepare":
        from class_project.dataset import prepare

        prepare()
    elif args.command == "train":
        from class_project.nlp import train

        train(args.seed, args.smoke)
    elif args.command == "evaluate":
        from class_project.nlp import evaluate

        evaluate(args.smoke, args.finbert)
    elif args.command in ["filter-news", "score-news", "snapshot", "compare"]:
        from class_project import financial

        if args.command == "filter-news":
            financial.filter_news(args.csv)
        elif args.command == "score-news":
            financial.score_news()
        elif args.command == "snapshot":
            financial.snapshot(args.sec_user_agent)
        else:
            financial.compare()
    elif args.command == "demo":
        from class_project.demo import launch

        launch(args.share, args.finbert)
    elif args.command == "inspect-data":
        directories = (
            [Path(args.path).expanduser()]
            if args.path
            else [ROOT / "data", output_path("financial")]
        )
        for directory in directories:
            if not directory.exists():
                print(directory.name, ": no local data found")
                continue
            for p in sorted(directory.rglob("*")):
                if p.is_file() and p.suffix in [".csv", ".parquet", ".duckdb", ".json"]:
                    print(p.relative_to(directory), p.stat().st_size, "bytes")
        print(
            "No database was opened or changed. For existing headline data, "
            "export ticker,published,headline into the class workspace; "
            "retain original publication times."
        )
    elif args.command == "export":
        from zipfile import ZIP_DEFLATED, ZipFile

        dest = output_path("exports", "class_training_artifacts.zip")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(dest, "w", ZIP_DEFLATED) as z:
            for p in ARTIFACTS.rglob("*"):
                relative = p.relative_to(ARTIFACTS)
                if (
                    p.is_file()
                    and relative.parts[0]
                    in ["dataset", "nlp", "evaluation", "financial", "submission"]
                    and "checkpoints" not in relative.parts
                ):
                    z.write(p, p.relative_to(ARTIFACTS).as_posix())
        print(dest)
    elif args.command == "writeup":
        from class_project.writeup import generate

        generate()


if __name__ == "__main__":
    main()
