"""Model registry: discover models in models_dir and lazy-load predictors."""
import json
from pathlib import Path
from typing import Any

from app.config import settings


def _model_batch_size(model_path: Path) -> int:
    """Read per-model batch_size from model_config.json if present."""
    config_file = model_path / "model_config.json"
    if config_file.is_file():
        try:
            data = json.loads(config_file.read_text(encoding="utf-8"))
            if isinstance(data.get("batch_size"), int) and data["batch_size"] > 0:
                return data["batch_size"]
        except (json.JSONDecodeError, OSError):
            pass
    return settings.default_batch_size


def _list_model_names() -> list[str]:
    """List subdirectory names in models_dir (each = one model)."""
    path = Path(settings.models_dir)
    if not path.is_dir():
        return []
    return [d.name for d in path.iterdir() if d.is_dir()]


class ModelRegistry:
    """Registry of available models; lazy-loads on first use."""

    def __init__(self) -> None:
        self._predictors: dict[str, Any] = {}
        self._batch_sizes: dict[str, int] = {}

    def model_names(self) -> list[str]:
        return _list_model_names()

    def get_batch_size(self, model_name: str, override: int | None = None) -> int:
        if override is not None:
            return override
        if model_name in self._batch_sizes:
            return self._batch_sizes[model_name]
        self._batch_sizes[model_name] = settings.default_batch_size
        return settings.default_batch_size

    def get_predictor(self, model_name: str):
        """Get or create predictor for model_name. Raises if model not found."""
        if model_name not in self._predictors:
            from app.inference.hf_predictor import load_hf_predictor

            path = Path(settings.models_dir) / model_name
            if not path.is_dir():
                raise FileNotFoundError(f"Model directory not found: {path}")
            self._predictors[model_name] = load_hf_predictor(path)
            self._batch_sizes[model_name] = _model_batch_size(path)
        return self._predictors[model_name]

    def predict(
        self,
        model_name: str,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[Any]:
        """Run prediction for model on texts, in chunks of batch_size."""
        predictor = self.get_predictor(model_name)
        size = self.get_batch_size(model_name, override=batch_size)
        return predictor.predict(texts, batch_size=size)


_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_predictor(model_name: str):
    return get_model_registry().get_predictor(model_name)
