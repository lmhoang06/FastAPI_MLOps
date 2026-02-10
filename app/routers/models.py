"""One route per model: POST /models/{model_name}/predict."""
from fastapi import APIRouter, HTTPException, status

from app.inference import get_model_registry
from app.schemas import BatchTextResponse, TextRequest, TextResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models():
    """List available model names (one route per model)."""
    registry = get_model_registry()
    return {"models": registry.model_names()}


@router.post("/{model_name}/predict", response_model=TextResponse | BatchTextResponse)
def predict(model_name: str, body: TextRequest):
    """
    Run model on single text or batch of texts.
    Uses model's default batch_size (or body.batch_size if set) when processing texts.
    """
    registry = get_model_registry()
    if model_name not in registry.model_names():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_name}. Available: {registry.model_names()}",
        )
    texts = [body.text] if body.text is not None else body.texts
    batch_size = body.batch_size
    results = registry.predict(model_name, texts=texts, batch_size=batch_size)
    if body.text is not None:
        return TextResponse(result=results[0])
    return BatchTextResponse(results=results)
