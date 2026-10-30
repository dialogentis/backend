"""Generate an ignored local configuration; refuse to overwrite existing credentials."""

from pathlib import Path
from secrets import token_hex

from cryptography.fernet import Fernet


def main() -> None:
    admin, migration, app, broker = (token_hex(24) for _ in range(4))
    values = {
        "APP_ENV": "development",
        "PUBLIC_BASE_URL": "http://localhost:18080",
        "POSTGRES_PASSWORD": admin,
        "MIGRATION_DB_PASSWORD": migration,
        "APP_DB_PASSWORD": app,
        "RABBITMQ_USER": "dialogentis",
        "RABBITMQ_PASSWORD": broker,
        "DATABASE_URL": f"postgresql://dialogentis_app:{app}@postgres:5432/dialogentis",
        "MIGRATION_DATABASE_URL": (
            f"postgresql://dialogentis_migrator:{migration}@postgres:5432/dialogentis"
        ),
        "BROKER_URL": f"amqp://dialogentis:{broker}@rabbitmq:5672/",
        "SESSION_SECRET": token_hex(32),
        "ENCRYPTION_KEY": Fernet.generate_key().decode(),
        "ENCRYPTION_KEY_VERSION": "1",
        "ARTIFACT_ROOT": "/app/artifacts",
        "SMTP_HOST": "mailpit",
        "SMTP_PORT": "1025",
        "SMTP_FROM": "no-reply@localhost",
    }
    target = Path(".env")
    descriptor = target.open("x")
    target.chmod(0o600)
    with descriptor:
        descriptor.write("".join(f"{key}={value}\n" for key, value in values.items()))
    print("Created .env (mode 0600). Values are intentionally not printed.")


if __name__ == "__main__":
    main()
