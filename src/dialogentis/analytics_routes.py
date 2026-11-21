from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time, timedelta
from math import ceil
from typing import Annotated, Literal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Integer,
    and_,
    case,
    cast,
    column,
    func,
    or_,
    select,
    values,
)

from dialogentis.analytics import LATENCY_BUCKETS_MS
from dialogentis.analytics_models import MetricHour
from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import InputModel
from dialogentis.dependencies import Database
from dialogentis.models import Agent, AgentDraft
from dialogentis.organizations import organization_context

router = APIRouter(prefix="/api/v1/organizations/{org_id}/analytics", tags=["analytics"])


class AnalyticsFilters(InputModel):
    start: date | None = Field(None, alias="from")
    end: date | None = Field(None, alias="to")
    time_zone: str = Field("UTC", min_length=1, max_length=80)
    channel: Literal["production", "api", "widget", "test", "all"] = "production"
    agent_id: UUID | None = None
    intent_scope: Literal["root", "active"] = "root"


class AnalyticsPoint(BaseModel):
    day: date
    dialogues: int = 0
    messages: int = 0


class AnalyticsRate(BaseModel):
    count: int
    total: int
    value: float | None


class AnalyticsLatency(BaseModel):
    samples: int
    p50_ms: int | None
    p95_ms: int | None


class AnalyticsIntent(BaseModel):
    agent_id: UUID
    agent_name: str | None
    intent_id: UUID
    intent_name: str | None
    count: int


class AnalyticsView(BaseModel):
    start: date
    end: date
    time_zone: str
    as_of: datetime
    dialogues: int
    messages: int
    fallback: AnalyticsRate
    errors: AnalyticsRate
    latency: AnalyticsLatency
    internal_latency: AnalyticsLatency
    external_http_latency: AnalyticsLatency
    latency_method: Literal["nearest_rank_bucket_upper_bound"] = "nearest_rank_bucket_upper_bound"
    series: list[AnalyticsPoint]
    intents: list[AnalyticsIntent]


def _intervals(start: date, end: date, zone: ZoneInfo):
    for offset in range((end - start).days + 1):
        day = start + timedelta(days=offset)
        low = datetime.combine(day, time.min, zone).astimezone(UTC)
        high = datetime.combine(day + timedelta(days=1), time.min, zone).astimezone(UTC)
        yield (
            day,
            low.replace(minute=0, second=0, microsecond=0),
            low.minute,
            high,
            high.replace(minute=0, second=0, microsecond=0),
            high.minute,
        )


def _histogram(totals: Counter[tuple[str, str]], metric: str) -> AnalyticsLatency:
    bins = [(bound, totals[(metric, str(bound))]) for bound in LATENCY_BUCKETS_MS]
    count = sum(samples for _, samples in bins) + totals[(metric, "overflow")]

    def percentile(quantile: float):
        rank = ceil(count * quantile)
        if rank == 0:
            return None
        for bound, samples in bins:
            rank -= samples
            if rank <= 0:
                return bound
        return None

    return AnalyticsLatency(samples=count, p50_ms=percentile(0.5), p95_ms=percentile(0.95))


def _rate(count: int, total: int) -> AnalyticsRate:
    return AnalyticsRate(count=count, total=total, value=count / total if total else None)


@router.get("", response_model=AnalyticsView)
async def analytics_report(
    org_id: UUID, query: Annotated[AnalyticsFilters, Query()], user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id)
    if (
        query.agent_id is not None
        and await db.scalar(
            select(Agent.id).where(Agent.organization_id == org_id, Agent.id == query.agent_id)
        )
        is None
    ):
        raise HTTPException(404, "Агент не найден")
    as_of = datetime.now(UTC)
    try:
        zone = ZoneInfo(query.time_zone)
        end = query.end or as_of.astimezone(zone).date()
        start = query.start or end - timedelta(days=6)
        if not 0 <= (end - start).days < 90:
            raise HTTPException(422, "Укажите период от 1 до 90 дней")
        intervals = list(_intervals(start, end, zone))
    except ValueError, ZoneInfoNotFoundError, OverflowError:
        raise HTTPException(422, "Некорректный период или часовой пояс") from None

    days = values(
        column("day", Date),
        column("begin_hour", DateTime(timezone=True)),
        column("begin_minute", Integer),
        column("end_at", DateTime(timezone=True)),
        column("end_hour", DateTime(timezone=True)),
        column("end_minute", Integer),
        name="analytics_days",
    ).data(intervals)
    hour = MetricHour.bucket_at
    minute_entries = func.jsonb_each_text(MetricHour.minute_counts).table_valued("key", "value")
    low = case((hour == days.c.begin_hour, days.c.begin_minute), else_=0)
    high = case((hour == days.c.end_hour, days.c.end_minute), else_=60)
    partial_count = (
        select(func.coalesce(func.sum(cast(minute_entries.c.value, BigInteger)), 0))
        .select_from(minute_entries)
        .where(
            cast(minute_entries.c.key, Integer) >= low, cast(minute_entries.c.key, Integer) < high
        )
        .correlate(MetricHour, days)
        .scalar_subquery()
    )
    count = case((and_(low == 0, high == 60), MetricHour.count), else_=partial_count)
    by_day = case((MetricHour.metric.in_(("dialogues", "messages")), days.c.day))
    by_agent = case((MetricHour.metric == "intent", MetricHour.active_agent_id))
    statement = (
        select(MetricHour.metric, MetricHour.item, by_day, by_agent, func.sum(count))
        .select_from(MetricHour)
        .join(days, and_(hour >= days.c.begin_hour, hour < days.c.end_at))
        .where(MetricHour.organization_id == org_id)
        .group_by(MetricHour.metric, MetricHour.item, by_day, by_agent)
        .having(func.sum(count) > 0)
    )
    if query.channel == "production":
        statement = statement.where(MetricHour.channel.in_(("api", "widget")))
    elif query.channel != "all":
        statement = statement.where(MetricHour.channel == query.channel)
    intent_filter = MetricHour.metric == "intent"
    if query.intent_scope == "root":
        statement = statement.where(
            or_(~intent_filter, MetricHour.active_agent_id == MetricHour.root_agent_id)
        )
    if query.agent_id is not None:
        is_root = MetricHour.root_agent_id == query.agent_id
        if query.intent_scope == "root":
            statement = statement.where(is_root)
        else:
            statement = statement.where(
                or_(
                    and_(~intent_filter, is_root),
                    and_(intent_filter, MetricHour.active_agent_id == query.agent_id),
                )
            )
    totals: Counter[tuple[str, str]] = Counter()
    series: dict[date, Counter[str]] = defaultdict(Counter)
    intents: list[tuple[int, UUID, UUID]] = []
    for metric, item, day, agent_id, total in await db.execute(statement):
        if metric == "intent":
            assert agent_id is not None
            intents.append((int(total), agent_id, UUID(item)))
        else:
            totals[(metric, item)] += int(total)
            if day is not None:
                series[day][metric] += int(total)
    top = sorted(intents, key=lambda item: (-item[0], item[1], item[2]))[:10]
    labels = {}
    if top:
        identities = {agent for _, agent, _ in top}
        metadata = await db.execute(
            select(Agent.id, Agent.name, AgentDraft.configuration["intents"])
            .outerjoin(
                AgentDraft,
                and_(
                    AgentDraft.agent_id == Agent.id,
                    AgentDraft.organization_id == Agent.organization_id,
                ),
            )
            .where(Agent.organization_id == org_id, Agent.id.in_(identities))
        )
        labels = {
            agent: (name, {item["id"]: item["name"] for item in items or []})
            for agent, name, items in metadata
        }
    return AnalyticsView(
        start=start,
        end=end,
        time_zone=zone.key,
        as_of=as_of,
        dialogues=totals[("dialogues", "")],
        messages=totals[("messages", "")],
        fallback=_rate(totals[("fallback", "")], totals[("nlu", "")]),
        errors=_rate(totals[("failed", "")], totals[("completed", "")]),
        latency=_histogram(totals, "latency_total"),
        internal_latency=_histogram(totals, "latency_internal"),
        external_http_latency=_histogram(totals, "latency_http"),
        series=[
            AnalyticsPoint(
                day=day, dialogues=series[day]["dialogues"], messages=series[day]["messages"]
            )
            for day in sorted(series)
        ],
        intents=[
            AnalyticsIntent(
                agent_id=agent,
                intent_id=intent,
                count=total,
                agent_name=labels.get(agent, (None, {}))[0],
                intent_name=labels.get(agent, (None, {}))[1].get(str(intent)),
            )
            for total, agent, intent in top
        ],
    )
