from functools import lru_cache
from pathlib import Path
from typing import Literal

from cryptography.fernet import Fernet
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    app_env: Literal["development", "test", "production"] = "development"
    public_base_url: str = "http://localhost:18080"
    database_url: SecretStr | None = None
    migration_database_url: SecretStr | None = None
    broker_url: SecretStr | None = None
    session_secret: SecretStr | None = None
    encryption_key: SecretStr | None = None
    encryption_key_version: int = 1
    artifact_root: Path = Path("artifacts")
    deletion_journal_root: Path = Path("deletion-journal")
    smtp_host: str | None = None
    smtp_port: int = 1025
    smtp_user: str | None = None
    smtp_password: SecretStr | None = None
    smtp_from: str = "Dialogentis <no-reply@localhost>"
    max_training_jobs: int = 1
    max_body_bytes: int = 10 * 1024 * 1024

    @model_validator(mode="after")
    def validate_environment(self) -> Settings:
        if self.encryption_key:
            try:
                Fernet(self.encryption_key.get_secret_value().encode())
            except (ValueError, TypeError) as exc:
                raise ValueError("ENCRYPTION_KEY must be a Fernet key") from exc
        if self.app_env == "production":
            required = (
                self.database_url,
                self.broker_url,
                self.session_secret,
                self.encryption_key,
                self.smtp_host,
            )
            if not all(required):
                raise ValueError(
                    "Production requires database, broker, session, encryption and SMTP"
                )
            if not self.public_base_url.startswith("https://"):
                raise ValueError("Production PUBLIC_BASE_URL must use HTTPS")
            if self.session_secret and len(self.session_secret.get_secret_value()) < 32:
                raise ValueError("SESSION_SECRET must contain at least 32 characters")
            if not self.artifact_root.is_absolute():
                raise ValueError("Production ARTIFACT_ROOT must be absolute")
            if not self.deletion_journal_root.is_absolute():
                raise ValueError("Production DELETION_JOURNAL_ROOT must be absolute")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
