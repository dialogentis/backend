import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any, cast
from uuid import uuid4

import pytest

from dialogentis.artifacts import ArtifactInfo, ArtifactIntegrityError, ArtifactStorage


def test_artifacts_are_atomic_private_scoped_and_checked(tmp_path):
    storage = ArtifactStorage(tmp_path)
    org, other, artifact = uuid4(), uuid4(), uuid4()
    info = storage.write(org, artifact, [b"hello", b" world"], max_bytes=11)
    assert info == ArtifactInfo(11, hashlib.sha256(b"hello world").hexdigest())
    assert storage.read(org, artifact, info, max_bytes=11) == b"hello world"
    assert (tmp_path / str(org)).stat().st_mode & 0o777 == 0o700
    assert (tmp_path / str(org) / str(artifact)).stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileNotFoundError):
        storage.read(other, artifact, info, max_bytes=11)
    with pytest.raises(FileExistsError):
        storage.write(org, artifact, [b"replacement"], max_bytes=11)
    assert storage.read(org, artifact, info, max_bytes=11) == b"hello world"
    (tmp_path / str(org) / str(artifact)).write_bytes(b"HELLO WORLD")
    with pytest.raises(ArtifactIntegrityError):
        storage.read(org, artifact, info, max_bytes=11)
    storage.delete(org, artifact)
    storage.delete(org, artifact)
    assert not (tmp_path / str(org) / str(artifact)).exists()


def test_failed_write_does_not_publish_or_leave_temporary_files(tmp_path):
    storage = ArtifactStorage(tmp_path)
    org, artifact = uuid4(), uuid4()
    with pytest.raises(ValueError):
        storage.write(org, artifact, [b"123", b"45"], max_bytes=4)
    assert list((tmp_path / str(org)).iterdir()) == []
    with pytest.raises(TypeError):
        storage.write(org, artifact, [b"123", cast(Any, "not bytes")], max_bytes=20)
    assert list((tmp_path / str(org)).iterdir()) == []


def test_paths_cannot_escape_using_symlinks_or_untrusted_identifiers(tmp_path):
    root = tmp_path / "root"
    external = tmp_path / "external"
    root.mkdir()
    external.mkdir()
    (external / "keep").write_text("safe")
    storage = ArtifactStorage(root)
    org, artifact = uuid4(), uuid4()
    (root / str(org)).symlink_to(external, target_is_directory=True)
    with pytest.raises(OSError):
        storage.write(org, artifact, [b"attack"], max_bytes=6)
    with pytest.raises(OSError):
        storage.purge_organization(org)
    assert (external / "keep").read_text() == "safe"
    with pytest.raises(TypeError):
        storage.write(cast(Any, "../external"), artifact, [b"attack"], max_bytes=6)


def test_read_rejects_symlinks_fifos_bad_metadata_and_oversize(tmp_path):
    storage = ArtifactStorage(tmp_path)
    org, artifact = uuid4(), uuid4()
    info = storage.write(org, artifact, [b"a"], max_bytes=1)
    path = tmp_path / str(org) / str(artifact)
    path.unlink()
    path.symlink_to(tmp_path / "secret")
    with pytest.raises(OSError):
        storage.read(org, artifact, info, max_bytes=1)
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ArtifactIntegrityError):
        storage.read(org, artifact, info, max_bytes=1)
    with pytest.raises(ArtifactIntegrityError):
        storage.read(org, artifact, ArtifactInfo(1, "bad"), max_bytes=1)
    with pytest.raises(ArtifactIntegrityError):
        storage.read(org, artifact, info, max_bytes=0)


def test_missing_root_is_not_silently_treated_as_empty(tmp_path):
    storage = ArtifactStorage(tmp_path / "unmounted")
    with pytest.raises(FileNotFoundError):
        storage.purge_organization(uuid4())
    with pytest.raises(FileNotFoundError):
        storage.delete(uuid4(), uuid4())


def test_write_rejects_a_tenant_directory_with_insecure_permissions(tmp_path):
    org = uuid4()
    directory = tmp_path / str(org)
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    with pytest.raises(PermissionError):
        ArtifactStorage(tmp_path).write(org, uuid4(), [b"secret"], max_bytes=6)
    assert list(directory.iterdir()) == []


def test_concurrent_finalization_never_replaces_a_published_file(tmp_path):
    storage = ArtifactStorage(tmp_path)
    org, artifact = uuid4(), uuid4()
    barrier = Barrier(2)

    def write(value):
        barrier.wait(timeout=2)
        try:
            return value, storage.write(org, artifact, [value], max_bytes=20)
        except FileExistsError:
            return value, None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, [b"first", b"second"]))
    winners = [(value, info) for value, info in results if info is not None]
    assert len(winners) == 1
    value, info = winners[0]
    assert info is not None and storage.read(org, artifact, info, max_bytes=20) == value
    assert len(list((tmp_path / str(org)).iterdir())) == 1
