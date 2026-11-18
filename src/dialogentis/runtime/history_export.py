import csv
import io
import json
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.artifact_service import ArtifactQuotaExceeded, reserve_artifact, write_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import Settings, get_settings
from dialogentis.datasets.csv_export import safe_cell
from dialogentis.db import unit_of_work
from dialogentis.jobs.lifecycle import Apply, Claim, PermanentJobError, RetryableJobError
from dialogentis.jobs.models import Job
from dialogentis.models import Organization, User
from dialogentis.organizations import organization_context
from dialogentis.runtime.history import read_messages
from dialogentis.runtime.history_export_models import HistoryExportTask, HistoryFormat
from dialogentis.runtime.history_filters import HistoryFilters, filtered_history, history_cutoff
from dialogentis.runtime.history_models import ConversationHistory, HistoryTurn
from dialogentis.runtime.receipts import ReceiptError, _key

MAX_EXPORT_BYTES = 10 * 1024 * 1024
MAX_EXPORT_ROWS = 10_000
MAGIC = b"DGE1"
ENVELOPE_BYTES = len(MAGIC) + 12 + 16
FIELDS = (
    "record_type",
    "generated_at",
    "filters",
    "store_message_text",
    "conversation_id",
    "agent_id",
    "version_id",
    "release_id",
    "channel_type",
    "status",
    "conversation_at",
    "last_activity_at",
    "turn_id",
    "sequence",
    "turn_at",
    "intent_id",
    "error_code",
    "user_text",
    "agent_messages",
    "trace",
)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class HistoryFile:
    def __init__(self, format: HistoryFormat, metadata: dict):
        self.content = io.BytesIO()
        self.csv = csv.writer(self) if format == "csv" else None
        self.first = True
        if self.csv is None:
            self.write(_json(metadata)[:-1] + ',"records":[')
        else:
            self.csv.writerow(FIELDS)
            self.row({"record_type": "export", **metadata})

    def write(self, text: str) -> int:
        encoded = text.encode("utf-8")
        if self.content.tell() + len(encoded) > MAX_EXPORT_BYTES:
            raise PermanentJobError("history_export_too_large")
        self.content.write(encoded)
        return len(text)

    def row(self, data: dict) -> None:
        if self.csv is None:
            if not self.first:
                self.write(",")
            self.write(_json(data))
            self.first = False
        else:
            cells = []
            for field in FIELDS:
                value = data.get(field)
                if isinstance(value, dict | list | tuple | bool):
                    value = _json(value)
                cells.append(safe_cell(str(value)) if value is not None else "")
            self.csv.writerow(cells)

    def finish(self) -> bytes:
        if self.csv is None:
            self.write("]}\n")
        return self.content.getvalue()


def _cipher(settings: Settings) -> AESGCM:
    return AESGCM(_key(settings, b"history-export-file"))


def seal_history_export(
    data: bytes, organization_id: UUID, job_id: UUID, settings: Settings
) -> bytes:
    nonce = os.urandom(12)
    return (
        MAGIC
        + nonce
        + _cipher(settings).encrypt(
            nonce,
            data,
            MAGIC + organization_id.bytes + job_id.bytes,
        )
    )


def open_history_export(
    data: bytes, organization_id: UUID, job_id: UUID, settings: Settings
) -> bytes:
    if data[:4] != MAGIC or not ENVELOPE_BYTES <= len(data) <= MAX_EXPORT_BYTES + ENVELOPE_BYTES:
        raise ReceiptError("invalid_history_export")
    try:
        return _cipher(settings).decrypt(
            data[4:16], data[16:], MAGIC + organization_id.bytes + job_id.bytes
        )
    except InvalidTag:
        raise ReceiptError("invalid_history_export") from None


async def permitted_org(db: AsyncSession, task: HistoryExportTask) -> Organization:
    actor = await db.get(User, task.requested_by)
    if actor is None or not actor.email_verified:
        raise PermanentJobError("history_export_access_changed")
    try:
        await organization_context(db, actor, task.organization_id, "manage_history")
    except HTTPException:
        raise PermanentJobError("history_export_access_changed") from None
    org = await db.get(Organization, task.organization_id)
    assert org is not None
    if org.settings_revision != task.settings_revision:
        raise PermanentJobError("history_export_settings_changed")
    return org


async def export_history_file(
    db: AsyncSession, task: HistoryExportTask, org: Organization, settings: Settings
) -> tuple[bytes, datetime | None]:
    filters = HistoryFilters.model_validate(task.filters)
    output = HistoryFile(
        task.format,
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "filters": task.filters,
            "store_message_text": org.store_message_text,
        },
    )
    query = (
        filtered_history(org, filters)
        .add_columns(HistoryTurn)
        .outerjoin(
            HistoryTurn,
            and_(
                HistoryTurn.organization_id == ConversationHistory.organization_id,
                HistoryTurn.conversation_id == ConversationHistory.id,
                HistoryTurn.created_at > history_cutoff(org),
            ),
        )
        .order_by(ConversationHistory.created_at, ConversationHistory.id, HistoryTurn.sequence)
        .limit(MAX_EXPORT_ROWS + 1)
    )
    retained_until = None
    rows = await db.stream(query.execution_options(yield_per=4))
    try:
        count = 0
        async for conversation, turn in rows:
            if count == MAX_EXPORT_ROWS:
                raise PermanentJobError("history_export_too_large")
            count += 1
            earliest = (
                min(conversation.last_activity_at, turn.created_at)
                if turn
                else conversation.last_activity_at
            )
            expires = earliest + timedelta(days=org.retention_days)
            retained_until = expires if retained_until is None else min(expires, retained_until)
            record = {
                "record_type": "turn" if turn else "conversation",
                "conversation_id": str(conversation.id),
                "agent_id": str(conversation.agent_id),
                "version_id": str(conversation.version_id),
                "release_id": str(conversation.release_id) if conversation.release_id else None,
                "channel_type": conversation.channel_type,
                "status": conversation.status,
                "conversation_at": conversation.created_at.isoformat(),
                "last_activity_at": conversation.last_activity_at.isoformat(),
            }
            if turn is not None:
                payload = read_messages(turn, settings) if org.store_message_text else None
                record.update(
                    turn_id=str(turn.turn_id),
                    sequence=turn.sequence,
                    turn_at=turn.created_at.isoformat(),
                    intent_id=str(turn.intent_id) if turn.intent_id else None,
                    error_code=turn.error_code,
                    user_text=payload.input_text if payload else None,
                    agent_messages=payload.messages if payload else (),
                    trace=turn.trace,
                )
            output.row(record)
    finally:
        await rows.close()
    return output.finish(), retained_until


async def prepare_history_export(
    factory: async_sessionmaker[AsyncSession], current: Claim
) -> Apply:
    settings = get_settings()
    async with unit_of_work(factory, organization_id=current.organization_id) as db:
        task = await db.get(HistoryExportTask, current.job_id)
        if task is None:
            raise PermanentJobError("history_export_unavailable")
        org = await permitted_org(db, task)
        try:
            content, retained_until = await export_history_file(db, task, org, settings)
            encrypted = seal_history_export(
                content, current.organization_id, current.job_id, settings
            )
            del content
        except ReceiptError:
            raise PermanentJobError("history_export_read_failed") from None

    async def apply(db: AsyncSession, job: Job) -> None:
        task = await db.get(HistoryExportTask, job.id)
        if task is None:
            raise PermanentJobError("history_export_unavailable")
        await permitted_org(db, task)
        now = await db.scalar(select(func.clock_timestamp()))
        assert now is not None
        if retained_until is not None and retained_until <= now:
            raise PermanentJobError("history_export_settings_changed")
        try:
            artifact = await reserve_artifact(
                db,
                current.organization_id,
                kind="temporary",
                max_bytes=len(encrypted),
                settings=settings,
                job_id=job.id,
            )
            await write_artifact(
                db,
                ArtifactStorage(settings.artifact_root),
                current.organization_id,
                artifact.id,
                [encrypted],
            )
        except ArtifactQuotaExceeded as exc:
            raise PermanentJobError("artifact_quota_exceeded") from exc
        except OSError as exc:
            raise RetryableJobError("Artifact storage unavailable") from exc
        completed = await db.scalar(select(func.clock_timestamp()))
        assert completed is not None
        artifact.expires_at = completed + timedelta(hours=1)
        deadline = completed + timedelta(minutes=15)
        task.artifact_id = artifact.id
        task.download_until = deadline if retained_until is None else min(deadline, retained_until)

    return apply
