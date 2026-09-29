"""Package the source as a private Kaggle dataset and push the GPU training kernel.

Why a dataset rather than a git clone: the repository is private, so a clone
needs a GitHub token, and putting one into a third-party notebook is worse than
shipping 320 KB of source. C-MAPSS is fetched inside the kernel from the public
NASA archive, which needs internet but no credentials, so nothing secret ever
leaves this machine.

    python scripts/launch_kaggle_training.py --dry-run   # package only
    python scripts/launch_kaggle_training.py             # package and push
    python scripts/launch_kaggle_training.py --status     # poll the run

CREDENTIALS -- and a scope limit worth knowing about. A ``KGAT_`` access token
uploads the dataset and pushes the kernel, but its scope stops short of the rest:

* ``kernels_list`` returns 401 and ``dataset_list_files`` returns 403, so
  ``--status`` queries ``kernels_status`` on the known slug rather than
  enumerating.
* More importantly, a kernel pushed under an access token **runs with neither
  the requested GPU nor the attached private dataset**, even though
  ``kernel-metadata.json`` carries ``enable_gpu: true`` and the correct
  ``dataset_sources``. The kernel then fails at
  ``/kaggle/input/twinmind-src`` not existing, with ``nvidia-smi`` absent.

A legacy API token carries the missing scope. Create one at
kaggle.com/settings/account -> "Create New Token", which downloads kaggle.json,
then either put its username/key into .env as KAGGLE_USERNAME / KAGGLE_KEY or
drop the file at ~/.kaggle/kaggle.json, and re-run this script. Nothing else
needs to change.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("launch_kaggle")

STAGING = PROJECT_ROOT / ".kaggle_staging"
SOURCE_DIR = STAGING / "src_dataset"
KERNEL_DIR = STAGING / "kernel"

#: Everything the kernel needs, and nothing else. No data/, models/, .env.
INCLUDE_TREES = ("src", "scripts", "configs")
INCLUDE_FILES = ("pyproject.toml", "requirements.txt", "README.md")

SUBSETS = ("FD002", "FD003", "FD004")

KERNEL_SOURCE = '''\
"""Kaggle GPU kernel: A1 on FD002, FD003 and FD004, three seeds each."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC = Path("/kaggle/input/twinmind-src")
WORK = Path("/kaggle/working/twinmind")
OUT = Path("/kaggle/working/out")

# nvidia-smi is absent entirely when no accelerator is attached, so this has to
# be guarded: an unguarded probe raises FileNotFoundError and kills the run
# before any training starts.
try:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
except FileNotFoundError:
    gpu = ""
print("accelerator:", gpu or "NONE (nvidia-smi absent)", flush=True)

if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(SRC, WORK)
os.chdir(WORK)

pip = [sys.executable, "-m", "pip", "install", "-q"]
subprocess.run([*pip, "-r", "requirements.txt"], check=False)
subprocess.run([*pip, "-e", "."], check=True)

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", device, flush=True)

subprocess.run([sys.executable, "scripts/download_data.py", "--dataset", "cmapss"], check=True)

for subset in {subsets!r}:
    started = time.time()
    print("=" * 60, subset, "=" * 60, flush=True)
    rc = subprocess.run(
        [sys.executable, "-u", "scripts/train_health.py",
         "--subset", subset, "--seeds", "3", "--device", device],
        check=False,
    ).returncode
    print(subset, "rc", rc, f"{{time.time() - started:.0f}}s", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
for name in ("runs.jsonl", "summary.md"):
    candidate = WORK / "reports" / "results" / name
    if candidate.exists():
        shutil.copy2(candidate, OUT / name)
for checkpoint in (WORK / "models" / "a1").glob("*.pt"):
    shutil.copy2(checkpoint, OUT / checkpoint.name)
for f in sorted(OUT.iterdir()):
    print(f.name, f.stat().st_size // 1024, "KB")
'''


def package(username: str) -> tuple[Path, Path]:
    """Stage the source dataset and the kernel, returning their directories."""
    if STAGING.exists():
        shutil.rmtree(STAGING)
    SOURCE_DIR.mkdir(parents=True)
    KERNEL_DIR.mkdir(parents=True)

    for tree in INCLUDE_TREES:
        shutil.copytree(
            PROJECT_ROOT / tree,
            SOURCE_DIR / tree,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pt", "*.ckpt"),
        )
    for name in INCLUDE_FILES:
        shutil.copy2(PROJECT_ROOT / name, SOURCE_DIR / name)

    leaked = [
        p.name
        for p in SOURCE_DIR.rglob("*")
        if p.is_file() and (p.name in {".env", "kaggle.json"} or p.suffix == ".pem")
    ]
    if leaked:  # pragma: no cover - guarded by construction
        raise RuntimeError(f"refusing to upload; secret-looking files staged: {leaked}")

    (SOURCE_DIR / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "twinmind-src",
                "id": f"{username}/twinmind-src",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    (KERNEL_DIR / "twinmind_a1_fd234.py").write_text(
        KERNEL_SOURCE.format(subsets=SUBSETS), encoding="utf-8", newline="\n"
    )
    (KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{username}/twinmind-a1-fd234",
                "title": "twinmind-a1-fd234",
                "code_file": "twinmind_a1_fd234.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [f"{username}/twinmind-src"],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    files = [p for p in SOURCE_DIR.rglob("*") if p.is_file()]
    logger.info(
        "staged %d source files (%.0f KB) and the kernel",
        len(files),
        sum(p.stat().st_size for p in files) / 1024,
    )
    return SOURCE_DIR, KERNEL_DIR


def _api() -> Any:
    """An authenticated client, reusing the resolver the downloader already uses."""
    from twinmind.data.kaggle_auth import authenticate

    api, resolved = authenticate()
    logger.info("authenticated via %s", resolved.source)
    return api


def push(username: str) -> int:
    """Upload the dataset and push the kernel. Returns a process exit code."""
    api = _api()
    try:
        api.dataset_create_version(
            str(SOURCE_DIR), version_notes="twinmind source", dir_mode="zip", quiet=True
        )
        logger.info("updated existing dataset %s/twinmind-src", username)
    except Exception:  # noqa: BLE001 - first push has no version to update
        try:
            api.dataset_create_new(str(SOURCE_DIR), public=False, quiet=True, dir_mode="zip")
            logger.info("created dataset %s/twinmind-src", username)
        except Exception as exc:  # noqa: BLE001 - surface the real cause
            logger.error("dataset upload failed: %s", str(exc)[:300])
            if "401" in str(exc):
                logger.error(
                    "401 on upload means a read-only credential. A KGAT_ access token "
                    "can download but cannot write; create a legacy API token at "
                    "kaggle.com/settings/account and set KAGGLE_USERNAME/KAGGLE_KEY."
                )
            return 1

    try:
        api.kernels_push(str(KERNEL_DIR))
    except Exception as exc:  # noqa: BLE001
        logger.error("kernel push failed: %s", str(exc)[:300])
        return 1

    logger.info("pushed; watch it at https://www.kaggle.com/code/%s/twinmind-a1-fd234", username)
    return 0


def status(username: str) -> int:
    """Print the kernel's current run status."""
    api = _api()
    try:
        result = api.kernels_status(f"{username}/twinmind-a1-fd234")
    except Exception as exc:  # noqa: BLE001
        logger.error("status failed: %s", str(exc)[:200])
        return 1
    logger.info("status: %s", getattr(result, "status", result))
    return 0


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Launch A1 GPU training on Kaggle.")
    parser.add_argument("--dry-run", action="store_true", help="Stage locally, do not upload.")
    parser.add_argument("--status", action="store_true", help="Poll the pushed kernel.")
    parser.add_argument("--username", default=None, help="Kaggle username; else from env.")
    args = parser.parse_args()

    import os

    from dotenv import load_dotenv

    load_dotenv()
    username = args.username or os.getenv("KAGGLE_USERNAME")
    if not username:
        logger.error("KAGGLE_USERNAME is not set; put it in .env or pass --username")
        return 1

    if args.status:
        return status(username)

    package(username)
    if args.dry_run:
        logger.info("dry run: staged under %s, nothing uploaded", STAGING)
        return 0
    return push(username)


if __name__ == "__main__":
    sys.exit(main())
