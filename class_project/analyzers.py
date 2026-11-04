"""Strict research backend: never substitute another model after load failures."""

from __future__ import annotations

from class_project.common import LABELS


class CheckpointAnalyzer:
    name = "class-distilbert"

    def __init__(self, checkpoint: str, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device).eval()
        )
        self.labels = [
            str(self.model.config.id2label[i]).lower() for i in range(self.model.config.num_labels)
        ]
        if set(self.labels) != set(LABELS):
            raise ValueError(f"Checkpoint must supply named financial labels: {self.labels}")

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        output = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if any(not text.strip() for text in batch):
                raise ValueError("Enter a nonempty financial sentence.")
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            with self.torch.inference_mode():
                values = (
                    self.model(**{k: v.to(self.device) for k, v in encoded.items()})
                    .logits.softmax(-1)
                    .cpu()
                    .tolist()
                )
            for row in values:
                probs = dict(zip(self.labels, row, strict=True))
                output.append(
                    {
                        "label": max(probs, key=probs.get),
                        "probabilities": probs,
                        "sentiment": probs["positive"] - probs["negative"],
                    }
                )
        return output

    def score_batch(self, texts: list[str]) -> list[float]:
        return [r["sentiment"] for r in self.predict_batch(texts)]

    def score(self, text: str) -> float:
        return self.score_batch([text])[0]

    def torn_verdict(self, text: str) -> bool:
        return False


def vader_predictions(texts: list[str]) -> list[dict]:
    from stock_monitor.sentiment import VaderAnalyzer

    analyzer = VaderAnalyzer()
    predictions = []
    for text in texts:
        score = analyzer.score(text)
        label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        predictions.append({"label": label, "sentiment": score})
    return predictions
