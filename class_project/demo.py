"""A focused classroom interface, with no live trading or news calls."""

from class_project.common import output_path, read_json
from class_project.sentence_analysis import analyze_document, apply_review_weights, display_rows


def launch(share=False, include_finbert=False):
    import gradio as gr

    from class_project.analyzers import CheckpointAnalyzer, vader_predictions

    own = CheckpointAnalyzer(str(output_path("nlp", "seed-42", "model")))
    reference = CheckpointAnalyzer("ProsusAI/finbert") if include_finbert else None

    def analyze(text):
        if not text.strip():
            return {"message": "Enter a financial sentence."}
        if len(text) > 4000:
            return {"message": "Please shorten the text to 4,000 characters."}
        result = {
            "vader": vader_predictions([text])[0],
            "our_distilbert": own.predict_batch([text])[0],
        }
        if reference:
            result["finbert_reference"] = reference.predict_batch([text])[0]
        for name, analyzer in [("our_distilbert", own), ("finbert_reference", reference)]:
            if analyzer:
                count = len(analyzer.tokenizer(text, truncation=False)["input_ids"])
                result[name]["input_tokens"] = count
                result[name]["truncated_at_128_tokens"] = count > 128
        result["note"] = (
            "Sentiment is not a stock forecast. Probabilities are classifier outputs, "
            "not calibrated certainty."
            " Transformer inputs beyond 128 tokens are truncated; VADER uses the full input."
        )
        return result

    def analyze_passage(body, headline, targets, mode):
        try:
            report = analyze_document(own, body, headline, targets, mode)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        weights = [[row["id"], 1.0] for row in report["sentences"]]
        return report, display_rows(report), weights, report

    def review_weights(report, weights, reason=""):
        try:
            revised = apply_review_weights(report, weights, reason)
        except (ValueError, TypeError) as error:
            raise gr.Error(str(error)) from error
        return revised, display_rows(revised), revised

    financial = output_path("financial", "metrics.json")
    evaluation = output_path("evaluation", "full", "metrics.json")
    with gr.Blocks(analytics_enabled=False) as app:
        gr.Markdown(
            "# Financial News Sentiment Study\nCompare language methods; "
            "then inspect one controlled financial application experiment."
        )
        with gr.Tab("Headline analysis"):
            text = gr.Textbox(label="Financial sentence or passage (raw model comparison)", lines=3)
            button = gr.Button("Compare sentiment")
            result = gr.JSON(label="Predictions")
            button.click(analyze, text, result)
            gr.Examples(
                [
                    ["The company raised its profit forecast."],
                    ["Revenue increased, but management warned of weaker demand."],
                    ["The company will release its results on Thursday."],
                ],
                text,
            )
        with gr.Tab("Sentence analysis + weights"):
            gr.Markdown(
                "## Inspect the evidence, then its weighted overall score\n"
                "Research-only extension using our **unchanged seed-42 DistilBERT**. "
                "Equal weighting is the baseline. Optional target matching gives explicitly "
                "matching units 2x weight and other context 1x; this is an unvalidated heuristic, "
                "not learned event importance. Headline status alone adds no extra weight. "
                "The score is not a probability or price forecast.\n\n"
                "**Calculation:** sentence score = P(positive) − P(negative); "
                "overall = sum(weight × sentence score) / sum(weights). "
                "Opposing scores are flagged even if their average is near zero."
            )
            passage = gr.Textbox(label="Financial passage", lines=6)
            headline = gr.Textbox(label="Optional original headline (not an instruction)")
            targets = gr.Textbox(
                label="Optional target aliases, comma-separated",
                placeholder="Nasdaq, QQQ, semiconductors",
            )
            mode = gr.Radio(
                ["Equal weights", "Target relevance (experimental)"],
                value="Equal weights",
                label="Weighting rule",
            )
            analyze_button = gr.Button("Analyze sentences and overall sentiment")
            stored = gr.State(None)
            sentence_table = gr.Dataframe(
                headers=[
                    "ID",
                    "Source",
                    "Text",
                    "Predicted class",
                    "Score",
                    "Base weight",
                    "Review multiplier",
                    "Normalized weight",
                    "Contribution",
                    "Truncated?",
                    "Weight reason",
                ],
                interactive=False,
                label="Evidence and contribution of every unit",
            )
            gr.Markdown(
                "**Optional manual review:** multiplier 1 keeps the automatic weight; "
                "2 doubles it; 0 excludes the unit. Allowed range: 0–5. Keep IDs unchanged. "
                "Document why you changed a weight; this changes interpretation, not the "
                "trained model. Weights cannot verify which news truly matters most. "
                "After editing the passage/target/mode, click Analyze again before reviewing."
            )
            weights = gr.Dataframe(
                headers=["Sentence ID", "Review multiplier"],
                datatype=["number", "number"],
                type="array",
                col_count=(2, "fixed"),
                interactive=True,
                label="Editable review weights for the displayed analysis",
            )
            reason = gr.Textbox(label="Optional reason for your manual weight changes", max_lines=3)
            review_button = gr.Button("Recalculate using reviewed weights")
            sentence_result = gr.JSON(
                label="Overall summary, probabilities, and method limitations"
            )
            analyze_button.click(
                analyze_passage,
                [passage, headline, targets, mode],
                [stored, sentence_table, weights, sentence_result],
            )
            review_button.click(
                review_weights,
                [stored, weights, reason],
                [stored, sentence_table, sentence_result],
            )
        with gr.Tab("Measured results"):
            gr.Markdown(
                "FinBERT previously trained on PhraseBank; its benchmark result is reference-only. "
                "Our financial results are prediction metrics, not trading returns."
            )
            gr.JSON(
                read_json(evaluation)
                if evaluation.exists()
                else {"status": "Full NLP evaluation not yet run."},
                label="Sentiment evaluation",
            )
            gr.JSON(
                read_json(financial)
                if financial.exists()
                else {"status": "Financial experiment pending; no improvement claim can be made."},
                label="LightGBM with/without sentiment",
            )
            sample = output_path("financial", "with_sentiment_final_predictions.csv")
            if sample.exists():
                import pandas as pd

                gr.Dataframe(
                    pd.read_csv(sample).head(1),
                    label="Historical example: date, raw probability and observed label",
                )
    app.launch(share=share)
