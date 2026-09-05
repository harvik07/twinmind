"""Download helpers: streaming fetch with retries, and manifest bookkeeping.

The CLI in ``scripts/download_data.py`` is a thin wrapper over this module so the
logic stays importable (and testable) from a notebook.

Idempotency contract: every dataset directory carries a ``manifest.json``
recording each file's name, size, sha256 and the fetch timestamp. A re-run
verifies checksums and skips the network entirely when they match.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

MANIFEST_NAME = "manifest.json"


class DownloadError(RuntimeError):
    """Raised when a dataset cannot be fetched or verified."""


@dataclass(frozen=True)
class FetchPolicy:
    """Network behaviour shared by every download."""

    timeout_seconds: float = 60.0
    max_retries: int = 3
    backoff_factor: float = 2.0
    chunk_bytes: int = 1024 * 1024


# --------------------------------------------------------------------------- #
# Checksums and manifests
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Streaming sha256 so a 200 MB archive never lands in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(directory: Path, files: list[Path]) -> dict[str, Any]:
    """Describe ``files`` (paths relative to ``directory``) for the manifest."""
    entries: list[dict[str, Any]] = [
        {
            "name": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    return {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_files": len(entries),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "files": entries,
    }


def write_manifest(directory: Path, files: list[Path]) -> dict[str, Any]:
    """Write ``manifest.json`` describing the dataset directory."""
    manifest = build_manifest(directory, files)
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_manifest(directory: Path) -> dict[str, Any] | None:
    """Read a manifest, or None when it is missing or unreadable."""
    path = directory / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    except (OSError, json.JSONDecodeError):
        logger.warning("manifest at %s is unreadable; treating as absent", path)
        return None


def manifest_is_satisfied(directory: Path) -> bool:
    """True when every file named in the manifest exists with a matching sha256."""
    manifest = read_manifest(directory)
    if not manifest or not manifest.get("files"):
        return False
    for entry in manifest["files"]:
        path = directory / str(entry["name"])
        if not path.exists() or path.stat().st_size != int(entry["size_bytes"]):
            return False
        if sha256_file(path) != entry["sha256"]:
            logger.warning("checksum mismatch for %s", path)
            return False
    return True


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def stream_download(url: str, destination: Path, policy: FetchPolicy, label: str = "") -> Path:
    """Stream ``url`` to ``destination`` with retries, backoff and a progress bar.

    Writes to a ``.part`` file and renames on success, so an interrupted download
    can never be mistaken for a complete one by the manifest check.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, policy.max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=policy.timeout_seconds) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", 0))
                with (
                    partial.open("wb") as fh,
                    tqdm(
                        total=total or None,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=label or destination.name,
                        leave=False,
                    ) as bar,
                ):
                    for chunk in response.iter_content(chunk_size=policy.chunk_bytes):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
            partial.replace(destination)
            return destination
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < policy.max_retries:
                delay = policy.backoff_factor ** (attempt - 1)
                logger.warning(
                    "download failed (attempt %d/%d): %s -- retrying in %.1fs",
                    attempt,
                    policy.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

    raise DownloadError(
        f"could not download {url} after {policy.max_retries} attempts: {last_error}"
    )


# --------------------------------------------------------------------------- #
# Archives
# --------------------------------------------------------------------------- #
def _is_within(base: Path, target: Path) -> bool:
    """Guard against zip entries that escape the extraction directory."""
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def safe_extract(archive: Path, destination: Path) -> list[Path]:
    """Extract a zip, refusing entries that would write outside ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = destination / member
            if not _is_within(destination, target):
                raise DownloadError(f"refusing unsafe zip entry {member!r} in {archive}")
        zf.extractall(destination)
        extracted = [destination / m for m in zf.namelist() if not m.endswith("/")]
    return [p for p in extracted if p.is_file()]


def extract_nested_zips(directory: Path, max_depth: int = 3) -> list[Path]:
    """Recursively unpack zips found under ``directory``.

    The NASA C-MAPSS archive contains a second zip inside the first, so a single
    ``extractall`` leaves the actual .txt files still packed.
    """
    for _ in range(max_depth):
        inner = [p for p in directory.rglob("*.zip") if p.is_file()]
        if not inner:
            break
        for archive in inner:
            logger.info("unpacking nested archive %s", archive.name)
            safe_extract(archive, archive.parent)
            archive.unlink()
    return sorted(p for p in directory.rglob("*") if p.is_file())


def flatten_into(directory: Path, patterns: list[str]) -> list[Path]:
    """Move files matching ``patterns`` from any subdirectory up into ``directory``."""
    moved: list[Path] = []
    for pattern in patterns:
        for path in sorted(directory.rglob(pattern)):
            if not path.is_file():
                continue
            target = directory / path.name
            if path != target:
                shutil.move(str(path), str(target))
            moved.append(target)
    # Drop the now-empty scaffolding the archives came with.
    for child in sorted(directory.rglob("*"), key=lambda p: -len(p.parts)):
        if child.is_dir() and not any(child.iterdir()):
            child.rmdir()
    return sorted(set(moved))
