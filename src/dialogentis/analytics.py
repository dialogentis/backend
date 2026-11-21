from collections import Counter
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Literal
from uuid import UUID

from sqlalchemy import BigInteger, delete, func, select
from sqlalchemy.dialects.postgresql import array, insert
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.analytics_models import MetricEvent, MetricHour
from dialogentis.runtime.core import RuntimeResult
from dialogentis.runtime.models import TestConversation
from dialogentis.versions.models import AgentVersion

RETENTION = timedelta(days=90)
LATENCY_BUCKETS_MS = (
    0,
    25,
    50,
    100,
    250,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    5000,
    10_000,
    20_000,
    30_000,
    60_000,
)
type CounterKey = tuple[UUID, str, str]


def _event_key(turn_id: UUID, stage: str) -> str:
    return sha256(b"dialogentis:metric:v1:" + stage.encode("ascii") + turn_id.bytes).hexdigest()


async def _increment(
    db: AsyncSession,
    row: TestConversation,
    turn_id: UUID,
    stage: str,
    accepted_at: datetime,
    counts: Counter[CounterKey],
) -> None:
    minute = accepted_at.astimezone(UTC).minute
    bucket_at = accepted_at.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    event = (
        insert(MetricEvent)
        .values(
            organization_id=row.organization_id,
            event_key=_event_key(turn_id, stage),
            bucket_at=bucket_at,
        )
        .on_conflict_do_nothing()
        .returning(MetricEvent.event_key)
    )
    if await db.scalar(event) is None:
        return
    channel = (
        "widget" if row.widget_id is not None else "api" if row.api_key_id is not None else "test"
    )
    minute_key = str(minute)
    mutation = insert(MetricHour).values(
        [
            {
                "organization_id": row.organization_id,
                "bucket_at": bucket_at,
                "root_agent_id": row.agent_id,
                "active_agent_id": agent_id,
                "channel": channel,
                "metric": metric,
                "item": item,
                "count": count,
                "minute_counts": {minute_key: count},
            }
            for (agent_id, metric, item), count in counts.items()
        ]
    )
    await db.execute(
        mutation.on_conflict_do_update(
            index_elements=[
                "organization_id",
                "bucket_at",
                "root_agent_id",
                "active_agent_id",
                "channel",
                "metric",
                "item",
            ],
            set_={
                "count": MetricHour.count + mutation.excluded.count,
                "minute_counts": func.jsonb_set(
                    MetricHour.minute_counts,
                    array([minute_key]),
                    func.to_jsonb(
                        func.coalesce(
                            MetricHour.minute_counts[minute_key].astext.cast(BigInteger), 0
                        )
                        + mutation.excluded.count
                    ),
                ),
            },
        )
    )


async def record_accepted(
    db: AsyncSession,
    row: TestConversation,
    turn_id: UUID,
    accepted_at: datetime,
    *,
    kind: Literal["dialogues", "messages"],
) -> None:
    await _increment(db, row, turn_id, kind, accepted_at, Counter({(row.agent_id, kind, ""): 1}))


async def record_completed(
    db: AsyncSession,
    row: TestConversation,
    turn_id: UUID,
    accepted_at: datetime,
    ended_at: datetime,
    result: RuntimeResult | None,
    external_http_ms: int = 0,
) -> None:
    scope = row.organization_id
    if await db.get(MetricEvent, (scope, _event_key(turn_id, "messages"))) is None:
        return
    agent_id = row.agent_id
    counts = Counter({(agent_id, "completed", ""): 1})
    if result is None or result.state.status == "failed":
        counts[(agent_id, "failed", "")] += 1
    if result is not None:
        total = max(0, round((ended_at - accepted_at).total_seconds() * 1_000))
        external = min(total, external_http_ms)
        for metric, duration in (
            ("latency_total", total),
            ("latency_internal", total - external),
            ("latency_http", external),
        ):
            bucket = next(
                (str(bound) for bound in LATENCY_BUCKETS_MS if duration <= bound), "overflow"
            )
            counts[(agent_id, metric, bucket)] += 1
        if result.nlu_observations:
            counts[(agent_id, "nlu", "")] += 1
        if any(event.event == "fallback" for event in result.trace):
            counts[(agent_id, "fallback", "")] += 1
        recognized = [item for item in result.nlu_observations if item.intent_id is not None]
        versions = {item.version_id for item in recognized}
        owners = {row.version_id: agent_id}
        if other_versions := versions - owners.keys():
            matched = await db.execute(
                select(AgentVersion.id, AgentVersion.agent_id).where(
                    AgentVersion.organization_id == scope, AgentVersion.id.in_(other_versions)
                )
            )
            owners.update({version_id: owner for version_id, owner in matched})
        for item in recognized:
            if active := owners.get(item.version_id):
                counts[(active, "intent", str(item.intent_id))] += 1
    await _increment(db, row, turn_id, "completed", accepted_at, counts)


async def cleanup_metrics(db: AsyncSession, organization_id: UUID, now: datetime) -> None:
    cutoff = (now.astimezone(UTC) - RETENTION).replace(minute=0, second=0, microsecond=0)
    for model in (MetricHour, MetricEvent):
        await db.execute(
            delete(model).where(model.organization_id == organization_id, model.bucket_at < cutoff)
        )
