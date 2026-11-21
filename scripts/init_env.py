"""Generate an ignored local configuration; refuse to overwrite existing credentials."""

import argparse
import os
from pathlib import Path
from secrets import token_hex

from cryptography.fernet import Fernet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".env"))
    target = parser.parse_args().output
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
        "DELETION_JOURNAL_ROOT": "/app/deletion-journal",
        "SMTP_HOST": "mailpit",
        "SMTP_PORT": "1025",
        "SMTP_FROM": "no-reply@localhost",
    }
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write("".join(f"{key}={value}\n" for key, value in values.items()))
    print(f"Created {target} (mode 0600).")


if __name__ == "__main__":
    main()
