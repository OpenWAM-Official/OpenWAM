"""Episode manifest format, validation and cache lookup for benchmark evaluation.

Benchmark-specific builders generate ordered entries; this module seals and
publishes them, then resolves complete manifests for eval submission. Workers load
an exact path and hash and select their episode range. Rebuilding the cache
updates its index without changing manifests already referenced by queued Tasks.

Manifests use JSON so submission and evaluator environments can share them without
loading each other's simulator or model dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Mapping


def _validate_json(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string object key at {path}: {key!r}")
            _validate_json(item, f"{path}.{key}")
        return
    raise TypeError(f"non-JSON value at {path}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return one strict, stable JSON representation."""

    _validate_json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def content_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and seal one benchmark-owned JSON manifest body."""

    value = dict(body)
    if "manifest_hash" in value:
        raise ValueError("manifest body must not contain manifest_hash")
    entries = value.get("entries")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("episode-manifest entries must be a list of objects")
    return value | {"manifest_hash": content_hash(value)}


def verify_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Validate a sealed manifest and return a plain detached dictionary."""

    value = dict(manifest)
    supplied_hash = value.pop("manifest_hash", None)
    actual_hash = content_hash(value)
    if supplied_hash != actual_hash:
        raise ValueError(f"episode-manifest hash mismatch: expected content hash {actual_hash}, got {supplied_hash}")
    if expected_manifest_hash is not None and supplied_hash != expected_manifest_hash:
        raise ValueError(f"unexpected episode-manifest hash: expected {expected_manifest_hash}, got {supplied_hash}")
    entries = value.get("entries")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("episode-manifest entries must be a list of objects")
    return value | {"manifest_hash": supplied_hash}


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    """Atomically create an immutable manifest file.

    Repeating the same write is idempotent.  Existing different content is
    rejected instead of silently mutating a manifest referenced by queued Tasks.
    """

    checked = verify_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(checked) + b"\n"
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to replace sealed episode manifest: {path}")
        return path
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def write_content_addressed_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    checked = verify_manifest(manifest)
    digest = checked["manifest_hash"].removeprefix("sha256:")
    return write_manifest(Path(root) / f"{digest}.json", checked)


def load_manifest(path: Path, **expected: Any) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read episode manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("episode manifest must be a JSON object")
    return verify_manifest(value, **expected)


def select_entries(manifest: Mapping[str, Any], start: int, count: int) -> list[dict[str, Any]]:
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("episode_start must be a non-negative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("num_episodes must be a positive integer")
    entries = manifest["entries"]
    if start + count > len(entries):
        raise ValueError(f"episode range {start}:{start + count} exceeds sealed manifest size {len(entries)}")
    return [dict(entry) for entry in entries[start : start + count]]


def manifest_index_path(root: str | Path, cache_key: Mapping[str, Any]) -> Path:
    """Locate an exact manifest request without scanning for a recent-looking file."""
    return Path(root) / "index" / (content_hash(cache_key).split(":")[1] + ".json")


def manifest_index_exists(root: str | Path, cache_key: Mapping[str, Any]) -> bool:
    """Return whether an index exists while exposing access errors."""

    path = manifest_index_path(root, cache_key)
    try:
        path.stat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"cannot read manifest cache index {path}: {exc}") from exc
    return True


def load_cached_manifest(root: str | Path, cache_key: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Resolve one cache pointer to its immutable manifest."""
    index_path = manifest_index_path(root, cache_key)
    try:
        pointer = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict) or set(pointer) != {"manifest_hash"}:
            raise ValueError("manifest cache index must contain only manifest_hash")
        manifest_hash = pointer.get("manifest_hash")
        if not isinstance(manifest_hash, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_hash) is None:
            raise ValueError("invalid manifest_hash; expected sha256 followed by 64 hexadecimal digits")
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read manifest cache index {index_path}: {exc}") from exc
    path = Path(root) / "manifests" / (manifest_hash.removeprefix("sha256:") + ".json")
    manifest = load_manifest(path, expected_manifest_hash=pointer["manifest_hash"])
    return path, manifest


def publish_manifest(root: str | Path, cache_key: Mapping[str, Any], manifest: Mapping[str, Any]) -> Path:
    """Publish a complete manifest without changing files bound to queued eval Tasks.

    Rebuilding replaces only the lookup pointer; previous content-addressed
    manifests remain available. Atomic pointer replacement prevents partial reads.
    """
    path = write_content_addressed_manifest(Path(root) / "manifests", manifest)
    checked = verify_manifest(manifest)
    index = manifest_index_path(root, cache_key)
    index.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=index.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump({"manifest_hash": checked["manifest_hash"]}, handle)
        os.replace(temporary, index)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def require_manifests(
    root: str | Path, cache_keys: Iterable[Mapping[str, Any]], command: str
) -> list[tuple[Path, dict[str, Any]]]:
    """Check every requested manifest before submitting any eval Tasks.

    Return manifests in request order, or report all failures together so users can
    build the missing cache in a separate manifest-building stage.
    """
    manifests, errors = [], []
    for cache_key in cache_keys:
        try:
            manifests.append(load_cached_manifest(root, cache_key))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{json.dumps(dict(cache_key), sort_keys=True)}: {exc}")
    if errors:
        raise ValueError(
            "Cannot load complete manifests:\n" + "\n".join(errors) + "\nTry building manifests: " + command
        )
    return manifests
