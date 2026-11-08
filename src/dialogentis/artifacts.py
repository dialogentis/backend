import hashlib
import hmac
import os
import re
import shutil
import stat
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4


class ArtifactIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactInfo:
    size: int
    sha256: str


class ArtifactStorage:
    def __init__(self, root: Path):
        self.root = root

    @contextmanager
    def _root(self) -> Generator[int]:
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            yield descriptor
        finally:
            os.close(descriptor)

    @contextmanager
    def _tenant(self, organization_id: UUID, *, create: bool = False) -> Generator[int]:
        if not isinstance(organization_id, UUID):
            raise TypeError("Organization ID must be a UUID")
        with self._root() as root:
            name = str(organization_id)
            if create:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=root)
                    os.fsync(root)
                except FileExistsError:
                    pass
            descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root)
            try:
                directory = os.fstat(descriptor)
                if directory.st_uid != os.geteuid() or directory.st_mode & 0o077:
                    raise PermissionError("Artifact directory is not private")
                yield descriptor
            finally:
                os.close(descriptor)

    def write(
        self, organization_id: UUID, artifact_id: UUID, chunks: Iterable[bytes], *, max_bytes: int
    ) -> ArtifactInfo:
        # The caller must hold the organization row lock and reject organizations being deleted.
        if not isinstance(artifact_id, UUID):
            raise TypeError("Artifact ID must be a UUID")
        if max_bytes < 0:
            raise ValueError("Invalid artifact size limit")
        with self._tenant(organization_id, create=True) as tenant:
            temporary = f".{uuid4()}.tmp"
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=tenant,
            )
            try:
                digest = hashlib.sha256()
                size = 0
                with os.fdopen(descriptor, "wb") as stream:
                    for chunk in chunks:
                        if not isinstance(chunk, bytes):
                            raise TypeError("Artifact chunks must be bytes")
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError("Artifact exceeds size limit")
                        digest.update(chunk)
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(
                    temporary,
                    str(artifact_id),
                    src_dir_fd=tenant,
                    dst_dir_fd=tenant,
                    follow_symlinks=False,
                )
                os.unlink(temporary, dir_fd=tenant)
                os.fsync(tenant)
                return ArtifactInfo(size=size, sha256=digest.hexdigest())
            finally:
                try:
                    os.unlink(temporary, dir_fd=tenant)
                except FileNotFoundError:
                    pass

    def read(
        self, organization_id: UUID, artifact_id: UUID, info: ArtifactInfo, *, max_bytes: int
    ) -> bytes:
        if not isinstance(artifact_id, UUID):
            raise TypeError("Artifact ID must be a UUID")
        if info.size < 0 or info.size > max_bytes or not re.fullmatch(r"[0-9a-f]{64}", info.sha256):
            raise ArtifactIntegrityError("Invalid artifact metadata")
        with self._tenant(organization_id) as tenant:
            descriptor = os.open(
                str(artifact_id), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=tenant
            )
            with os.fdopen(descriptor, "rb") as stream:
                file_stat = os.fstat(stream.fileno())
                if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != info.size:
                    raise ArtifactIntegrityError("Artifact size mismatch")
                content = stream.read(info.size + 1)
        if len(content) != info.size or not hmac.compare_digest(
            hashlib.sha256(content).hexdigest(), info.sha256
        ):
            raise ArtifactIntegrityError("Artifact checksum mismatch")
        return content

    def delete(self, organization_id: UUID, artifact_id: UUID) -> None:
        if not isinstance(organization_id, UUID) or not isinstance(artifact_id, UUID):
            raise TypeError("Organization and artifact IDs must be UUIDs")
        with self._root() as root:
            try:
                tenant = os.open(
                    str(organization_id), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root
                )
            except FileNotFoundError:
                return
            try:
                try:
                    os.unlink(str(artifact_id), dir_fd=tenant)
                    os.fsync(tenant)
                except FileNotFoundError:
                    pass
            finally:
                os.close(tenant)

    def purge_organization(self, organization_id: UUID) -> None:
        if not isinstance(organization_id, UUID):
            raise TypeError("Organization ID must be a UUID")
        with self._root() as root:
            try:
                shutil.rmtree(str(organization_id), dir_fd=root)
                os.fsync(root)
            except FileNotFoundError:
                pass
