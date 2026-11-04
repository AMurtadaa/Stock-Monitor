"""Build an explicitly marked draft from actual saved experiment results."""

from class_project.common import output_path, read_json, write_json


def generate():
    nlp = read_json(output_path("evaluation", "full", "metrics.json"))
    financial_path = output_path("financial", "metrics.json")
    financial = read_json(financial_path) if financial_path.exists() else None
    lines = [
        "# Financial Sentiment and Stock Prediction",
        "",
        "**Evidence-based draft: student review and formatting required before submission.**",
        "",
        "## Page 1: Problem, data and existing application",
        "We test financial sentiment classification and a controlled application to Stock-Monitor. "
        "The sentiment study uses the 75%-agreement Financial PhraseBank subset, "
        "cleaned and split 70/15/15. Stock-Monitor supplies the finance-enhanced VADER baseline, "
        "financial feature builder and LightGBM model.",
        "",
        "## Page 2: Language model and sentiment results",
        "DistilBERT is fine-tuned for three sentiment classes for three epochs, "
        "with development macro-F1 selecting the checkpoint. "
        "Seeds 42, 17 and 73 share fixed splits. "
        "The seed-42 run is selected in advance "
        "for the application experiment.",
        "",
        "| Model | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ]
    for name, result in nlp["models"].items():
        lines.append(f"| {name} | {result['accuracy']:.4f} | {result['macro_f1']:.4f} |")
    lines += [
        "",
        "The FinBERT reference is not an unseen-data benchmark: "
        "it was previously trained on Financial PhraseBank.",
        "Review the saved confusion matrices and 15 selected examples; "
        "add explanations for at least five errors.",
        "",
        "## Page 3: Financial application",
        "Two raw-probability LightGBM classifiers use identical monthly examples and parameters; "
        "one excludes sentiment. The target is 12-month outperformance versus SPY. "
        "Training label-end dates are purged before evaluation.",
        "",
    ]
    if financial:
        lines += [
            "| Variant | Test Brier (lower better) | Test ROC-AUC | Test accuracy |",
            "|---|---:|---:|---:|",
        ]
        for name, result in financial["models"].items():
            m = result["metrics"]["final"]
            lines.append(
                f"| {name} | {m['brier']:.4f} | {m['roc_auc']:.4f} | {m['accuracy']:.4f} |"
            )
    else:
        lines += [
            "**Financial comparison not yet run. Do not claim improved financial prediction.**"
        ]
    lines += [
        "",
        "## Page 4: Limitations, lessons and references",
        "Discuss small samples; differences between financial sentences and live headlines; "
        "pretrained model exposure; fixed surviving-company panel; overlapping annual labels; "
        "historical timestamp uncertainty; and zero sentiment on days without news. "
        "A 48-hour lag reduces timestamp risk but cannot prove archive availability. "
        "Probability outputs are not promises, and these are prediction experiments "
        "rather than trading-return tests.",
        "",
        "References: Hugging Face text-classification workflow; "
        "Financial PhraseBank (Malo et al., 2014); "
        "DistilBERT model card; ProsusAI FinBERT model card; FNSPID; upstream Stock-Monitor.",
        "",
        "Add your contribution statement and course-required disclosure of AI assistance.",
    ]
    dest = output_path("submission", "report_draft.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output_path("submission", "results_summary.json"), {"nlp": nlp, "financial": financial}
    )
    print("Wrote report_draft.md and results_summary.json. Review before submission.")
