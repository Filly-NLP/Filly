from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_NAME: str = "FILLY"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=True, validation_alias="FILLY_DEBUG")

    # Database
    DATABASE_URL: str = Field(default="sqlite:///./filly.db", validation_alias="DATABASE_URL")

    # CORS
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"],
        validation_alias="CORS_ORIGINS",
    )

    # NLP inference
    MAX_INPUT_LENGTH: int = Field(default=5000, ge=1, validation_alias="MAX_INPUT_LENGTH")
    GECTOR_ITERATIONS: int = Field(default=5, ge=1, validation_alias="GECTOR_ITERATIONS")
    DEVICE: str = Field(default="auto", validation_alias="DEVICE")
    ROBERTA_MODEL: str = Field(
        default="jcblaise/roberta-tagalog-large",
        validation_alias="ROBERTA_MODEL",
    )

    # Paths. These fields can be overridden with matching environment variables.
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    NORMALIZATION_DATA_DIR: Path = BASE_DIR / "normalization" / "data"
    NORMALIZER_RULES_PATH: Path = Field(
        default=BASE_DIR / "artifacts" / "normalizer" / "rules.json",
        validation_alias="NORMALIZER_RULES_PATH",
    )
    NORMALIZER_VOCAB_PATH: Path = Field(
        default=BASE_DIR / "artifacts" / "normalizer" / "vocabulary.txt",
        validation_alias="NORMALIZER_VOCAB_PATH",
    )
    GECTOR_MODEL_PATH: Path = Field(
        default=BASE_DIR.parent / "best.pt",
        validation_alias="GECTOR_MODEL_PATH",
    )

    model_config = SettingsConfigDict(
        env_prefix="FILLY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
