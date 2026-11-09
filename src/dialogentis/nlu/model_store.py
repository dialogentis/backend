"""Authenticated internal model artifacts; never sign uploaded model data."""

import asyncio
import base64
import hashlib
import hmac
import json
import pickle
import struct
import sys
from dataclasses import dataclass
from importlib.metadata import version
from platform import python_version
from uuid import UUID

import numpy
import pydantic
import scipy
import sklearn
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.artifact_records import Artifact
from dialogentis.artifact_service import read_artifact, write_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import Settings
from dialogentis.nlu.adapter import NLUAdapter

MAGIC = b"DIALOGENTIS-NLU\x01"
MAX_HEADER_BYTES = 64 * 1024
MAC_BYTES = hashlib.sha256().digest_size


class InvalidModelArtifact(ValueError):
    pass


@dataclass(frozen=True)
class ModelIdentity:
    organization_id: UUID
    agent_id: UUID
    version_id: UUID
    artifact_id: UUID


def _key(settings: Settings) -> bytes:
    if settings.encryption_key is None or settings.encryption_key_version < 1:
        raise InvalidModelArtifact("Model signing key unavailable")
    encoded = settings.encryption_key.get_secret_value().encode("ascii")
    try:
        Fernet(encoded)
        secret = base64.urlsafe_b64decode(encoded)
    except (ValueError, TypeError) as exc:
        raise InvalidModelArtifact("Model signing key unavailable") from exc
    return hmac.digest(secret, b"dialogentis:nlu:artifact-signing:v1", "sha256")


def _environment() -> dict[str, str]:
    return {
        "implementation": "dialogentis-nlu-1",
        "python": python_version(),
        "sklearn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pydantic": pydantic.__version__,
        "scipy": scipy.__version__,
        "python_crfsuite": version("python-crfsuite"),
        "google_re2": version("google-re2"),
        "byteorder": sys.byteorder,
        "pointer_bits": str(struct.calcsize("P") * 8),
    }


def _identity(identity: ModelIdentity) -> dict[str, str]:
    return {
        "organization_id": str(identity.organization_id),
        "agent_id": str(identity.agent_id),
        "version_id": str(identity.version_id),
        "artifact_id": str(identity.artifact_id),
    }


def pack_model(adapter: NLUAdapter, identity: ModelIdentity, settings: Settings) -> bytes:
    if type(adapter) is not NLUAdapter or adapter.version_id != identity.version_id:
        raise InvalidModelArtifact("Model identity mismatch")
    key = _key(settings)
    # Only platform-trained models are passed here; this function is not an upload endpoint.
    payload = pickle.dumps(adapter, protocol=pickle.HIGHEST_PROTOCOL)
    header = {
        "schema_version": 1,
        "identity": _identity(identity),
        "key_version": settings.encryption_key_version,
        "environment": _environment(),
        "payload_size": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "intent_manifest": adapter.intent_model.manifest,
        "entity_manifest": adapter.entity_model.manifest if adapter.entity_model else None,
    }
    try:
        header_bytes = json.dumps(
            header, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise InvalidModelArtifact("Invalid model manifest") from exc
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise InvalidModelArtifact("Model manifest exceeds size limit")
    prefix = MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes
    if len(prefix) + len(payload) + MAC_BYTES > settings.max_model_artifact_bytes:
        raise InvalidModelArtifact("Model artifact exceeds size limit")
    signature = hmac.new(key, prefix, "sha256")
    signature.update(payload)
    return prefix + payload + signature.digest()


def unpack_model(content: bytes, identity: ModelIdentity, settings: Settings) -> NLUAdapter:
    key = _key(settings)
    fixed = len(MAGIC) + 4
    if (
        len(content) < fixed + MAC_BYTES
        or len(content) > settings.max_model_artifact_bytes
        or not content.startswith(MAGIC)
    ):
        raise InvalidModelArtifact("Invalid model artifact")
    header_size = struct.unpack(">I", content[len(MAGIC) : fixed])[0]
    body_start = fixed + header_size
    if header_size > MAX_HEADER_BYTES or body_start > len(content) - MAC_BYTES:
        raise InvalidModelArtifact("Invalid model artifact")
    signature = hmac.new(key, memoryview(content)[:-MAC_BYTES], "sha256").digest()
    if not hmac.compare_digest(signature, content[-MAC_BYTES:]):
        raise InvalidModelArtifact("Model authentication failed")
    try:
        header = json.loads(content[fixed:body_start])
    except (ValueError, UnicodeDecodeError) as exc:
        raise InvalidModelArtifact("Invalid model manifest") from exc
    payload = memoryview(content)[body_start:-MAC_BYTES]
    if (
        not isinstance(header, dict)
        or header.get("schema_version") != 1
        or header.get("identity") != _identity(identity)
        or header.get("key_version") != settings.encryption_key_version
        or header.get("environment") != _environment()
        or header.get("payload_size") != len(payload)
        or header.get("payload_sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise InvalidModelArtifact("Model identity or compatibility mismatch")
    try:
        # The MAC and identity are verified before this executable format is loaded.
        adapter = pickle.loads(payload)
    except Exception as exc:
        raise InvalidModelArtifact("Model could not be loaded") from exc
    if (
        type(adapter) is not NLUAdapter
        or adapter.version_id != identity.version_id
        or adapter.intent_model.manifest != header.get("intent_manifest")
        or (adapter.entity_model.manifest if adapter.entity_model else None)
        != header.get("entity_manifest")
    ):
        raise InvalidModelArtifact("Model manifest mismatch")
    return adapter


async def store_model(
    db: AsyncSession,
    storage: ArtifactStorage,
    identity: ModelIdentity,
    adapter: NLUAdapter,
    settings: Settings,
) -> Artifact:
    content = await asyncio.to_thread(pack_model, adapter, identity, settings)
    return await write_artifact(
        db,
        storage,
        identity.organization_id,
        identity.artifact_id,
        [content],
        expected_kind="model",
    )


async def load_model(
    db: AsyncSession, storage: ArtifactStorage, identity: ModelIdentity, settings: Settings
) -> NLUAdapter:
    content = await read_artifact(
        db,
        storage,
        identity.organization_id,
        identity.artifact_id,
        max_bytes=settings.max_model_artifact_bytes,
        expected_kind="model",
    )
    return await asyncio.to_thread(unpack_model, content, identity, settings)
