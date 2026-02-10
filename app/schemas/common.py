"""Request/response schemas for text-in, result-out models."""
from typing import Any

from pydantic import BaseModel, model_validator


class TextRequest(BaseModel):
    """Single text or batch of texts. Exactly one of text or texts must be set."""

    text: str | None = None
    texts: list[str] | None = None
    batch_size: int | None = None  # override model default for this request

    @model_validator(mode="after")
    def one_of_text_or_texts(self) -> "TextRequest":
        if self.text is not None and self.texts is not None:
            raise ValueError("Provide either 'text' or 'texts', not both.")
        if self.text is None and (self.texts is None or len(self.texts) == 0):
            raise ValueError("Provide either 'text' or non-empty 'texts'.")
        return self


class TextResponse(BaseModel):
    """Single prediction result (e.g. label + score)."""

    result: Any  # label, score, or dict per model type


class BatchTextResponse(BaseModel):
    """Batch prediction results."""

    results: list[Any]
