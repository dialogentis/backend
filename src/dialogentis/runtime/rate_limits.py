from datetime import datetime, timedelta
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.auth.security import digest
from dialogentis.channels.widget_access import WidgetAccess
from dialogentis.models import Base


class RateBucket(Base):
    __tablename__ = "runtime_rate_buckets"
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    subject: Mapped[str] = mapped_column(String(64), primary_key=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


async def limit_api_turn(db: AsyncSession, org_id: UUID, key_id: UUID, *, start: bool) -> None:
    kind = "start" if start else "send"
    for subject, rate in ((f"{kind}:organization", 10), (f"{kind}:key:{key_id}", 5)):
        await take_slot(db, org_id, subject, 1 / rate, 20)


async def limit_widget_turn(db: AsyncSession, access: WidgetAccess, *, start: bool) -> None:
    kind = "start" if start else "send"
    period, burst = (3, 5) if start else (0.2, 10)
    for subject in (
        f"widget:{kind}:{access.id}",
        digest(f"widget:{kind}:{access.id}:{access.client_ip}"),
    ):
        await take_slot(db, access.organization_id, subject, period, burst)


async def take_slot(
    db: AsyncSession, org_id: UUID, subject: str, seconds: float, burst: int
) -> None:
    now = func.statement_timestamp()
    period = timedelta(seconds=seconds)
    updated = await db.scalar(
        insert(RateBucket)
        .values(organization_id=org_id, subject=subject, available_at=now + period)
        .on_conflict_do_update(
            index_elements=[RateBucket.organization_id, RateBucket.subject],
            set_={"available_at": func.greatest(RateBucket.available_at, now) + period},
            where=RateBucket.available_at <= now + (burst - 1) * period,
        )
        .returning(RateBucket.subject)
    )
    if updated is None:
        raise HTTPException(
            429, "Превышен лимит запросов", headers={"Retry-After": str(max(1, int(seconds)))}
        )
