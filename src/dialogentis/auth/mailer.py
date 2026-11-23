import asyncio
import json
import smtplib
import ssl
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import and_, delete, or_, select, update

from dialogentis.auth.models import AuthToken, EmailOutbox, RateEvent, Session
from dialogentis.config import Settings, get_settings
from dialogentis.db import session_factory, unit_of_work

MAX_DELIVERY_ATTEMPTS = 5


def send_email(config: Settings, payload: dict) -> None:
    if not config.smtp_host:
        raise ValueError("SMTP_HOST is required")
    message = EmailMessage()
    message["From"], message["To"], message["Subject"] = (
        config.smtp_from,
        payload["to"],
        payload["subject"],
    )
    message.set_content(payload["text"])
    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=10) as smtp:
        if config.app_env == "production":
            smtp.starttls(context=ssl.create_default_context())
        if config.smtp_user:
            smtp.login(
                config.smtp_user,
                config.smtp_password.get_secret_value() if config.smtp_password else "",
            )
        smtp.send_message(message)


async def dispatch_once(config: Settings, factory) -> int:
    if config.encryption_key is None or not config.smtp_host:
        raise ValueError("SMTP and encryption settings are required")
    sent = 0
    async with unit_of_work(factory) as db:
        now = datetime.now(UTC)
        rows = (
            await db.scalars(
                select(EmailOutbox)
                .where(
                    EmailOutbox.sent_at.is_(None),
                    EmailOutbox.available_at <= now,
                    EmailOutbox.attempts < MAX_DELIVERY_ATTEMPTS,
                )
                .order_by(EmailOutbox.available_at)
                .limit(10)
                .with_for_update(skip_locked=True)
            )
        ).all()
        for row in rows:
            row.attempts += 1
            try:
                payload = json.loads(
                    Fernet(config.encryption_key.get_secret_value().encode()).decrypt(
                        row.encrypted_payload.encode()
                    )
                )
                await asyncio.to_thread(send_email, config, payload)
            except smtplib.SMTPException, OSError, InvalidToken, ValueError:
                if row.attempts == MAX_DELIVERY_ATTEMPTS:
                    row.available_at, row.encrypted_payload = now, ""
                else:
                    row.available_at = now + timedelta(seconds=60 * 2**row.attempts)
            else:
                row.sent_at, row.encrypted_payload = now, ""
                sent += 1
        cutoff = now - timedelta(days=1)
        await db.execute(delete(RateEvent).where(RateEvent.created_at < cutoff))
        await db.execute(delete(AuthToken).where(AuthToken.expires_at < now))
        await db.execute(delete(Session).where(Session.expires_at < now))
        failed_mail = and_(
            EmailOutbox.sent_at.is_(None), EmailOutbox.attempts >= MAX_DELIVERY_ATTEMPTS
        )
        await db.execute(
            delete(EmailOutbox).where(
                or_(
                    EmailOutbox.sent_at < cutoff,
                    and_(failed_mail, EmailOutbox.available_at < cutoff),
                )
            )
        )
        await db.execute(
            update(EmailOutbox)
            .where(failed_mail, EmailOutbox.encrypted_payload != "")
            .values(encrypted_payload="")
        )
    return sent


async def main():
    config = get_settings()
    if config.database_url is None:
        raise ValueError("DATABASE_URL is required")
    factory = session_factory(config.database_url.get_secret_value())
    try:
        while True:
            await dispatch_once(config, factory)
            await asyncio.sleep(1)
    finally:
        await factory.kw["bind"].dispose()


if __name__ == "__main__":
    asyncio.run(main())
