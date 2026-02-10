"""FastAPI app: multiple routes for multiple models (1 route per model)."""
from fastapi import FastAPI

from app.routers import models_router

app = FastAPI(
    title="Multi-Model API",
    description="One route per model; each accepts text or batch of texts.",
)

app.include_router(models_router)
