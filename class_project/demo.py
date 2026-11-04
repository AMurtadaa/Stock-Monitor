"""A focused classroom interface, with no live trading or news calls."""

from class_project.common import output_path, read_json


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
        result["note"] = (
            "Sentiment is not a stock forecast. Probabilities are classifier outputs, "
            "not calibrated certainty."
        )
        return result

    financial = output_path("financial", "metrics.json")
    evaluation = output_path("evaluation", "full", "metrics.json")
    with gr.Blocks() as app:
        gr.Markdown(
            "# Financial News Sentiment Study\nCompare language methods; "
            "then inspect one controlled financial application experiment."
        )
        with gr.Tab("Headline analysis"):
            text = gr.Textbox(label="Financial sentence", lines=3)
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
