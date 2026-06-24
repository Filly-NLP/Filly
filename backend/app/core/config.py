from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # App
    APP_NAME: str = "FILLY"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "sqlite:///./filly.db"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # NLP
    MAX_WORDS: int = 250
    ROBERTA_MODEL: str = "jcblaise/roberta-tagalog-large"
    USE_GPU: bool = True

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    NORMALIZATION_DATA_DIR: Path = BASE_DIR / "normalization" / "data"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
