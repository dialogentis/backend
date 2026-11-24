"""Authentication for immutable dialogue snapshots."""

import base64
import hmac
import json
from dataclasses import dataclass
from uuid import UUID

from cryptography.fernet import Fernet

from dialogentis.config import Settings


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


def _identity(identity: ModelIdentity) -> dict[str, str]:
    return {
        "organization_id": str(identity.organization_id),
        "agent_id": str(identity.agent_id),
        "version_id": str(identity.version_id),
        "artifact_id": str(identity.artifact_id),
    }


def pack_dialogue_manifest(identity: ModelIdentity, config_hash: str, settings: Settings) -> bytes:
    prefix = b"DIALOGENTIS-DIALOGUE\x01"
    content = json.dumps(
        {
            **_identity(identity),
            "config_sha256": config_hash,
            "key_version": settings.encryption_key_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    signature = hmac.digest(_key(settings), prefix + content, "sha256")
    return prefix + signature


def verify_dialogue_manifest(
    content: bytes, identity: ModelIdentity, config_hash: str, settings: Settings
) -> None:
    if not hmac.compare_digest(content, pack_dialogue_manifest(identity, config_hash, settings)):
        raise InvalidModelArtifact("Dialogue manifest authentication failed")
