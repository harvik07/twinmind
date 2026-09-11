"""Fetch the raw datasets into data/raw/<dataset>/.

Idempotent: each dataset directory gets a manifest.json of names, sizes, sha256
and a fetch timestamp. A re-run whose checksums match does no network I/O and
exits 0. Use --force to re-download regardless.

    python scripts/download_data.py --dataset all
    python scripts/download_data.py --dataset cmapss --force
    python scripts/download_data.py --dataset all --include-optional
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import cast

from twinmind.data import kaggle_auth
from twinmind.data.download import (
    DownloadError,
    FetchPolicy,
    extract_nested_zips,
    flatten_into,
    manifest_is_satisfied,
    safe_extract,
    stream_download,
    write_manifest,
)
from twinmind.utils.config import DataConfig, load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import raw_dir

logger = get_logger("download_data")

DATASETS = ["cmapss", "ai4i", "favorita", "skab"]
OPTIONAL = {"skab"}


def _policy(config: DataConfig) -> FetchPolicy:
    return FetchPolicy(
        timeout_seconds=config.download.timeout_seconds,
        max_retries=config.download.max_retries,
        backoff_factor=config.download.backoff_factor,
        chunk_bytes=config.download.chunk_bytes,
    )


# --------------------------------------------------------------------------- #
# C-MAPSS
# --------------------------------------------------------------------------- #
def fetch_cmapss(config: DataConfig, directory: Path) -> list[Path]:
    """Download the NASA archive and unpack both zip levels."""
    archive = directory / "cmapss_source.zip"
    stream_download(config.cmapss.url, archive, _policy(config), label="C-MAPSS")
    safe_extract(archive, directory)
    archive.unlink(missing_ok=True)
    extract_nested_zips(directory)

    wanted = [
        f"{kind}_{subset}.txt"
        for kind in ("train", "test", "RUL")
        for subset in config.cmapss.subsets
    ]
    files = flatten_into(directory, wanted)
    found = {p.name for p in files}
    missing = sorted(set(wanted) - found)
    if missing:
        raise DownloadError(f"C-MAPSS archive did not yield expected files: {missing}")

    # The archive also ships the damage-propagation paper and a readme describing
    # the operating regimes and fault modes. Keep them beside the data (the data
    # card cites them) rather than scattered in the archive's own folder names.
    docs = directory / "docs"
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in found and path.name != "manifest.json":
            docs.mkdir(exist_ok=True)
            path.replace(docs / path.name)
    for child in sorted(directory.rglob("*"), key=lambda p: -len(p.parts)):
        if child.is_dir() and child != docs and not any(child.iterdir()):
            child.rmdir()

    return [p for p in files if p.name in set(wanted)] + sorted(docs.glob("*"))


# --------------------------------------------------------------------------- #
# AI4I 2020
# --------------------------------------------------------------------------- #
def fetch_ai4i(config: DataConfig, directory: Path) -> list[Path]:
    """Fetch AI4I 2020, preferring ucimlrepo and falling back to the UCI zip."""
    target = directory / "ai4i2020.csv"
    try:
        from ucimlrepo import fetch_ucirepo

        logger.info("fetching AI4I 2020 via ucimlrepo (id=%d)", config.ai4i.uci_id)
        repo = fetch_ucirepo(id=config.ai4i.uci_id)
        frame = repo.data.original
        frame.to_csv(target, index=False)
        return [target]
    except Exception as exc:  # noqa: BLE001 - any ucimlrepo failure falls back to the zip
        logger.warning("ucimlrepo failed (%s); falling back to the UCI archive", exc)

    archive = directory / "ai4i_source.zip"
    stream_download(config.ai4i.csv_url, archive, _policy(config), label="AI4I 2020")
    try:
        extracted = safe_extract(archive, directory)
    except zipfile.BadZipFile:
        # The URL may serve the CSV directly rather than a zip.
        archive.replace(target)
        return [target]
    archive.unlink(missing_ok=True)
    csvs = [p for p in extracted if p.suffix.lower() == ".csv"]
    if not csvs:
        raise DownloadError("AI4I archive contained no CSV")
    if csvs[0] != target:
        csvs[0].replace(target)
    return [target]


# --------------------------------------------------------------------------- #
# Favorita (Kaggle)
# --------------------------------------------------------------------------- #
def fetch_favorita(config: DataConfig, directory: Path) -> list[Path]:
    """Download the Kaggle competition files.

    Authentication is delegated to :mod:`twinmind.data.kaggle_auth`, which
    resolves the credential scheme, normalises the value and maps 401 and 403
    onto separate, actionable errors. Every kaggle import is deferred until
    after .env has been loaded, because the package authenticates at import.
    """
    slug = config.favorita.kaggle_competition
    api, resolved = kaggle_auth.authenticate()

    logger.info("downloading Kaggle competition %s", slug)
    try:
        api.competition_download_files(slug, path=str(directory), quiet=False)
    except Exception as exc:  # noqa: BLE001 - the client raises bare API exceptions
        code = kaggle_auth.status_code(exc)
        if code == 403:
            raise kaggle_auth.KaggleForbiddenError(slug, resolved.source, str(exc)[:200]) from exc
        if code == 401:
            raise kaggle_auth.KaggleUnauthorizedError(resolved.source, str(exc)[:200]) from exc
        raise DownloadError(f"Kaggle download failed: {str(exc)[:300]}") from exc

    archives = sorted(directory.glob("*.zip"))
    for archive in archives:
        safe_extract(archive, directory)
        archive.unlink()

    files = sorted(p for p in directory.glob("*.csv"))
    expected = {
        "train.csv",
        "test.csv",
        "stores.csv",
        "oil.csv",
        "holidays_events.csv",
        "transactions.csv",
    }
    missing = sorted(expected - {p.name for p in files})
    if missing:
        raise DownloadError(f"Kaggle download is missing expected files: {missing}")
    return files


# --------------------------------------------------------------------------- #
# SKAB (optional)
# --------------------------------------------------------------------------- #
def fetch_skab(config: DataConfig, directory: Path) -> list[Path]:
    """Fetch the SKAB CSVs over raw.githubusercontent (no git clone)."""
    import requests

    policy = _policy(config)
    files: list[Path] = []
    api = "https://api.github.com/repos/waico/SKAB/contents/data"

    for group in config.skab.groups:
        response = requests.get(f"{api}/{group}", timeout=policy.timeout_seconds)
        if response.status_code != 200:
            logger.warning("could not list SKAB group %r (HTTP %d)", group, response.status_code)
            continue
        target_dir = directory / group
        target_dir.mkdir(parents=True, exist_ok=True)
        for entry in response.json():
            if entry.get("type") != "file" or not entry["name"].endswith(".csv"):
                continue
            url = f"{config.skab.base_url}/data/{group}/{entry['name']}"
            files.append(
                stream_download(url, target_dir / entry["name"], policy, label=f"skab/{group}")
            )

    if not files:
        raise DownloadError("no SKAB CSVs were retrieved")
    return files


FETCHERS = {
    "cmapss": fetch_cmapss,
    "ai4i": fetch_ai4i,
    "favorita": fetch_favorita,
    "skab": fetch_skab,
}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def fetch_dataset(name: str, config: DataConfig, force: bool = False) -> bool:
    """Fetch one dataset. Returns True if the network was used, False if skipped."""
    directory = raw_dir() / name
    directory.mkdir(parents=True, exist_ok=True)

    if not force and manifest_is_satisfied(directory):
        logger.info(
            "%s: already present and checksums match -- nothing to do (use --force to refetch)",
            name,
        )
        return False

    logger.info("%s: fetching into %s", name, directory)
    files = FETCHERS[name](config, directory)
    summary = write_manifest(directory, files)
    logger.info(
        "%s: %d files, %.1f MB, manifest written",
        name,
        summary["n_files"],
        summary["total_bytes"] / 1e6,
    )
    return True


def _check_auth(config: DataConfig) -> int:
    """Authenticate and read one competition's metadata, downloading nothing."""
    slug = config.favorita.kaggle_competition
    try:
        info = kaggle_auth.check_auth(slug)
    except kaggle_auth.KaggleAuthError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Kaggle authentication OK")
    for field in (
        "username",
        "auth_method",
        "source",
        "client_version",
        "supports_access_token",
    ):
        logger.info("  %-22s %s", field, info[field])
    found = info["competitions_found"]
    logger.info("  %-22s %s", "competitions matching", found or "(none listed)")
    if not any(slug in name for name in found):
        logger.warning(
            "%r was not in the listing. Credentials work, but you may not have "
            "accepted its rules yet: %s",
            slug,
            kaggle_auth.COMPETITION_URL.format(slug=slug),
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Download the raw datasets.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
        help="Which dataset to fetch. 'all' skips optional datasets unless "
        "--include-optional is given.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if checksums match.")
    parser.add_argument(
        "--check-auth",
        action="store_true",
        help="Verify Kaggle credentials and list one competition, without downloading.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional datasets (skab) when --dataset all.",
    )
    return parser


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args()
    config = cast(DataConfig, load_config("data"))

    if args.check_auth:
        return _check_auth(config)

    if args.dataset == "all":
        targets = [d for d in DATASETS if d not in OPTIONAL or args.include_optional]
    else:
        targets = [args.dataset]

    failures: list[str] = []
    for name in targets:
        try:
            fetch_dataset(name, config, force=args.force)
        except (DownloadError, kaggle_auth.KaggleAuthError) as exc:
            optional = name in OPTIONAL
            logger.error(
                "%s: %s\n%s", name, "optional dataset failed" if optional else "FAILED", exc
            )
            if not optional:
                failures.append(name)

    if failures:
        logger.error("failed datasets: %s", ", ".join(failures))
        return 1
    logger.info("done: %s", ", ".join(targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
