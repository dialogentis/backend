import json

from cryptography.fernet import Fernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.auth.models import EmailOutbox


async def has_untagged_invitation_mail(db: AsyncSession, key: str) -> bool:
    rows = (
        await db.scalars(
            select(EmailOutbox.encrypted_payload).where(
                EmailOutbox.organization_id.is_(None), EmailOutbox.sent_at.is_(None)
            )
        )
    ).all()
    cipher = Fernet(key.encode())
    return any(
        json.loads(cipher.decrypt(encrypted.encode())).get("subject") == "Приглашение в Dialogentis"
        for encrypted in rows
    )
