"""App settings: models directory, default batch size."""
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    models_dir: Path = Path(__file__).resolve().parent.parent / "models"
    default_batch_size: int = 8

    class Config:
        env_prefix = "APP_"
        env_file = ".env"


settings = Settings()
