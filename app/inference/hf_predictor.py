"""HuggingFace-based predictor for models in models/ (e.g. DistilBERT)."""
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class HFPredictor:
    """Loads a HuggingFace sequence classification model and runs inference in batches."""

    def __init__(self, model_path: Path, device: str | None = None) -> None:
        self.model_path = Path(model_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = getattr(
            self.model.config, "id2label", {}
        ) or getattr(self.model.config, "id2label", {})

    def predict(self, texts: list[str], batch_size: int = 8) -> list[Any]:
        """Predict for each text; returns list of dicts with label and score."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
            with torch.no_grad():
                out = self.model(**encoded)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1)
            for j in range(len(batch)):
                pred_idx = logits[j].argmax().item()
                score = probs[j][pred_idx].item()
                label = self.id2label.get(str(pred_idx), f"LABEL_{pred_idx}")
                results.append({"label": label, "score": score})
        return results


def load_hf_predictor(model_path: Path) -> HFPredictor:
    return HFPredictor(model_path)
