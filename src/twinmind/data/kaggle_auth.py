"""Kaggle authentication: resolution, capability detection and a clear error taxonomy.

Why this module exists
----------------------
Kaggle changed its credential scheme. There are now two:

* **Legacy API key** -- ``KAGGLE_USERNAME`` plus a 32-character hex ``KAGGLE_KEY``,
  sent as HTTP Basic auth. This is what ``~/.kaggle/kaggle.json`` contains.
* **Access token** -- a single ``KGAT_``-prefixed value, supplied as
  ``KAGGLE_API_TOKEN`` or in ``~/.kaggle/access_token``.

Client support differs by version: the token scheme arrived in ``kaggle`` 2.x.
On a 1.x client a ``KGAT_`` token placed in ``KAGGLE_KEY`` is sent as a Basic-auth
password, which the server rejects with **401** -- an error that looks like a bad
password rather than a scheme mismatch. That is the trap this module removes.

Two further hazards it handles:

* ``kaggle/__init__.py`` calls ``api.authenticate()`` at *import* time. Importing
  the package before ``.env`` has been loaded means the credentials are simply
  absent. Every import here is therefore deferred until after ``load_dotenv()``.
* ``.env`` values routinely arrive wrapped in quotes or padded with whitespace.
  Every value is normalised before use.

Security
--------
No function here logs, prints or writes a credential. Where a value must be
referred to, :func:`redact` shows its first four characters and length only.
``test_no_secret_reaches_the_logs`` enforces that.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

#: Environment variable holding a single access token (kaggle >= 2.x).
ENV_API_TOKEN = "KAGGLE_API_TOKEN"
#: Legacy pair.
ENV_USERNAME = "KAGGLE_USERNAME"
ENV_KEY = "KAGGLE_KEY"

#: Access tokens carry this prefix. Legacy keys are 32 hex characters.
TOKEN_PREFIX = "KGAT_"

SETTINGS_URL = "https://www.kaggle.com/settings"
COMPETITION_URL = "https://www.kaggle.com/competitions/{slug}"

AuthScheme = Literal["access_token", "legacy_api_key"]


# --------------------------------------------------------------------------- #
# Errors -- one per actionable cause
# --------------------------------------------------------------------------- #
class KaggleAuthError(RuntimeError):
    """Base class for every Kaggle authentication failure."""


class KaggleCredentialsNotFoundError(KaggleAuthError):
    """No credentials in any supported location."""

    def __init__(self, checked: list[str]) -> None:
        locations = "\n".join(f"    - {c}" for c in checked)
        super().__init__(
            "NO KAGGLE CREDENTIALS FOUND.\n\n"
            f"Checked, in order:\n{locations}\n\n"
            "To fix, put ONE of these in .env at the repository root:\n\n"
            f"    {ENV_API_TOKEN}=KGAT_...        (preferred; a single access token)\n"
            "  or\n"
            f"    {ENV_USERNAME}=your-username\n"
            f"    {ENV_KEY}=<32-character legacy key>\n\n"
            f"Generate either at {SETTINGS_URL} -> API.\n"
            '"Generate New Token" yields the KGAT_ access token; the older\n'
            '"Create New Token" downloads a kaggle.json holding the legacy pair.'
        )


class KaggleUnauthorizedError(KaggleAuthError):
    """Credentials were found but the server rejected them (HTTP 401)."""

    def __init__(self, source: str, detail: str = "") -> None:
        super().__init__(
            "KAGGLE 401 UNAUTHORIZED -- credentials were found but rejected.\n\n"
            f"    source: {source}\n\n"
            "The value is present but not valid. Most likely it has been revoked,\n"
            "has expired, or was copied incompletely.\n\n"
            f"Regenerate it at {SETTINGS_URL} -> API, then update .env.\n"
            "Note that an access token (KGAT_...) and a legacy key are different\n"
            f"schemes: a KGAT_ value belongs in {ENV_API_TOKEN}, not {ENV_KEY}."
            + (f"\n\nServer detail: {detail}" if detail else "")
        )


class KaggleForbiddenError(KaggleAuthError):
    """Credentials are valid but the competition rules have not been accepted (HTTP 403)."""

    def __init__(self, slug: str, source: str, detail: str = "") -> None:
        super().__init__(
            "KAGGLE 403 FORBIDDEN -- your credentials are valid, but this\n"
            "competition's rules have not been accepted by your account.\n\n"
            f"    source:      {source}\n"
            f"    competition: {slug}\n\n"
            "An API token alone is never sufficient for competition data. Open\n\n"
            f"    {COMPETITION_URL.format(slug=slug)}\n\n"
            'sign in, and click "Join Competition" / "I Understand and Accept".\n'
            "Then re-run this command." + (f"\n\nServer detail: {detail}" if detail else "")
        )


# --------------------------------------------------------------------------- #
# Value hygiene
# --------------------------------------------------------------------------- #
def normalise_value(raw: str | None) -> str | None:
    """Strip whitespace and one layer of surrounding quotes.

    ``.env`` files routinely contain ``KAGGLE_KEY="abc123"`` or a value with a
    trailing space. Both are invisible in a terminal and produce a 401.
    """
    if raw is None:
        return None
    value = raw.strip().lstrip("﻿")  # also drop a stray BOM
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value or None


def redact(value: str | None) -> str:
    """A safe reference to a secret: first four characters and length only."""
    if not value:
        return "<empty>"
    return f"{value[:4]}...({len(value)} chars)"


def looks_like_access_token(value: str | None) -> bool:
    """True when a value carries the access-token prefix rather than a legacy key."""
    return bool(value) and str(value).startswith(TOKEN_PREFIX)


# --------------------------------------------------------------------------- #
# Client capability
# --------------------------------------------------------------------------- #
def kaggle_client_version() -> str:
    """Installed ``kaggle`` version, or ``"unknown"``."""
    try:
        from importlib.metadata import version

        return version("kaggle")
    except Exception:  # noqa: BLE001 - absence must not be fatal here
        return "unknown"


def client_supports_access_token() -> bool:
    """Whether the installed client understands the ``KGAT_`` access-token scheme.

    Determined by inspecting the installed module for the symbol that implements
    it, rather than by comparing version strings -- the check then stays honest
    if the feature moves.
    """
    try:
        from kaggle.api import kaggle_api_extended
    except Exception:  # noqa: BLE001 - import problems are reported by the caller
        return False
    return hasattr(kaggle_api_extended.KaggleApi, "_authenticate_with_access_token")


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolvedCredentials:
    """Credentials plus the *name* of where they came from. Never logged in full."""

    scheme: AuthScheme
    source: str
    username: str | None = None
    key: str | None = None
    api_token: str | None = None

    def describe(self) -> str:
        """A log-safe one-line description."""
        if self.scheme == "access_token":
            return f"{self.source} (access token {redact(self.api_token)})"
        return f"{self.source} (user {self.username!r}, key {redact(self.key)})"

    def as_environment(self) -> dict[str, str]:
        """The environment the kaggle client should see for this scheme."""
        if self.scheme == "access_token":
            return {ENV_API_TOKEN: self.api_token or ""}
        return {ENV_USERNAME: self.username or "", ENV_KEY: self.key or ""}


def _read_kaggle_json(path: Path) -> tuple[str | None, str | None]:
    """Read ``username`` and ``key`` from a kaggle.json, tolerating a BOM."""
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None, None
    return normalise_value(payload.get("username")), normalise_value(payload.get("key"))


def resolve_credentials(
    env: dict[str, str] | None = None,
    kaggle_json: Path | None = None,
    access_token_file: Path | None = None,
) -> ResolvedCredentials:
    """Find credentials, in the documented precedence order.

    1. ``KAGGLE_API_TOKEN`` from the environment (when the client supports it)
    2. ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` from the environment
    3. ``~/.kaggle/kaggle.json``

    A ``KGAT_`` value found in ``KAGGLE_KEY`` is recognised and routed as an
    access token, because that is the single most likely way to end up with a
    401: the two schemes look interchangeable and are not.

    Raises:
        KaggleCredentialsNotFoundError: nothing usable in any location.
    """
    environ = os.environ if env is None else env
    home = Path.home()
    kaggle_json = kaggle_json or home / ".kaggle" / "kaggle.json"
    access_token_file = access_token_file or home / ".kaggle" / "access_token"

    supports_token = client_supports_access_token()
    checked: list[str] = []

    # 1. Access token from the environment.
    checked.append(f"${ENV_API_TOKEN} (environment)")
    token = normalise_value(environ.get(ENV_API_TOKEN))
    if token:
        if supports_token:
            return ResolvedCredentials(
                scheme="access_token", source=f"${ENV_API_TOKEN}", api_token=token
            )
        logger.warning(
            "%s is set but the installed kaggle client (%s) has no access-token "
            "support; ignoring it and falling back to the legacy key. Upgrade with "
            "'pip install \"kaggle>=2.2\"' to use it.",
            ENV_API_TOKEN,
            kaggle_client_version(),
        )

    # 2. Legacy pair from the environment -- with KGAT_ misplacement detection.
    checked.append(f"${ENV_USERNAME} + ${ENV_KEY} (environment)")
    username = normalise_value(environ.get(ENV_USERNAME))
    key = normalise_value(environ.get(ENV_KEY))

    if looks_like_access_token(key):
        if supports_token:
            logger.info(
                "the value in $%s is an access token (%s), not a legacy key; using it as one",
                ENV_KEY,
                redact(key),
            )
            return ResolvedCredentials(
                scheme="access_token",
                source=f"${ENV_KEY} (access token)",
                api_token=key,
            )
        logger.error(
            "$%s holds an access token (%s) but the installed kaggle client (%s) "
            "only supports the legacy username+key scheme. It will be sent as a "
            "Basic-auth password and rejected with 401. Upgrade with "
            "'pip install \"kaggle>=2.2\"', or use a legacy key from %s.",
            ENV_KEY,
            redact(key),
            kaggle_client_version(),
            SETTINGS_URL,
        )

    if username and key:
        return ResolvedCredentials(
            scheme="legacy_api_key", source="environment", username=username, key=key
        )

    # 3. Access-token file, then kaggle.json.
    checked.append(str(access_token_file))
    if supports_token and access_token_file.exists():
        stored = normalise_value(access_token_file.read_text(encoding="utf-8-sig"))
        if stored:
            return ResolvedCredentials(
                scheme="access_token", source=str(access_token_file), api_token=stored
            )

    checked.append(str(kaggle_json))
    if kaggle_json.exists():
        file_user, file_key = _read_kaggle_json(kaggle_json)
        if looks_like_access_token(file_key) and supports_token:
            return ResolvedCredentials(
                scheme="access_token",
                source=f"{kaggle_json} (access token)",
                api_token=file_key,
            )
        if file_user and file_key:
            return ResolvedCredentials(
                scheme="legacy_api_key",
                source=str(kaggle_json),
                username=file_user,
                key=file_key,
            )

    raise KaggleCredentialsNotFoundError(checked)


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
def status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction across the client's exception shapes."""
    for attribute in ("status", "code", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None and isinstance(getattr(response, "status_code", None), int):
        code: int = response.status_code
        return code
    text = str(exc)
    for candidate in (401, 403):
        if f"{candidate}" in text:
            return candidate
    return None


def authenticate(credentials: ResolvedCredentials | None = None) -> tuple[Any, ResolvedCredentials]:
    """Authenticate and return ``(api, credentials)``.

    The kaggle package is imported here rather than at module scope: importing it
    triggers an immediate ``authenticate()`` inside its own ``__init__``, so it
    must not happen until ``.env`` has been loaded and the environment prepared.

    Raises:
        KaggleCredentialsNotFoundError: nothing usable was found.
        KaggleUnauthorizedError: credentials found but rejected.
    """
    resolved = credentials or resolve_credentials()
    logger.info("kaggle auth source: %s", resolved.describe())

    # Present exactly the scheme we resolved, so a stale variable of the other
    # kind cannot win inside the client's own precedence.
    for name in (ENV_API_TOKEN, ENV_USERNAME, ENV_KEY):
        os.environ.pop(name, None)
    os.environ.update(resolved.as_environment())

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - kaggle is a hard dependency
        raise KaggleAuthError(
            "the 'kaggle' package is not installed; run: pip install -e \".[dev]\""
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as exc:
        # kaggle 2.x prints its own help and calls exit(1) when nothing works.
        raise KaggleUnauthorizedError(resolved.source, "client rejected all methods") from exc
    except Exception as exc:  # noqa: BLE001 - normalised into the taxonomy below
        if status_code(exc) == 401:
            raise KaggleUnauthorizedError(resolved.source, str(exc)[:200]) from exc
        raise KaggleAuthError(f"Kaggle authentication failed: {str(exc)[:300]}") from exc

    if resolved.scheme == "access_token" and not api.config_values.get("username"):
        raise KaggleUnauthorizedError(resolved.source, "token did not resolve to a user")

    logger.info(
        "authenticated as %r via %s",
        api.config_values.get("username"),
        api.config_values.get("auth_method", resolved.scheme),
    )
    return api, resolved


def check_auth(slug: str) -> dict[str, Any]:
    """Authenticate and read one competition's metadata without downloading.

    Returns a small dict describing what worked, suitable for printing.
    """
    api, resolved = authenticate()
    username = api.config_values.get("username")
    method = str(api.config_values.get("auth_method", resolved.scheme))

    try:
        listed = api.competitions_list(search=slug)
    except Exception as exc:  # noqa: BLE001 - mapped into the taxonomy
        code = status_code(exc)
        if code == 403:
            raise KaggleForbiddenError(slug, resolved.source, str(exc)[:200]) from exc
        if code == 401:
            raise KaggleUnauthorizedError(resolved.source, str(exc)[:200]) from exc
        raise KaggleAuthError(f"could not list competitions: {str(exc)[:300]}") from exc

    names = _competition_names(listed)
    return {
        "username": username,
        "auth_method": method,
        "source": resolved.source,
        "client_version": kaggle_client_version(),
        "supports_access_token": client_supports_access_token(),
        "competitions_found": names[:5],
    }


def _competition_names(listed: Any) -> list[str]:
    """Pull competition refs out of whatever shape the client returned.

    kaggle 1.x returns a list; 2.x returns a response object wrapping one.
    """
    candidates = listed
    if not isinstance(listed, list):
        for attribute in ("competitions", "items", "results"):
            value = getattr(listed, attribute, None)
            if isinstance(value, list):
                candidates = value
                break
        else:
            candidates = []
    names: list[str] = []
    for item in candidates:
        ref = getattr(item, "ref", None) or getattr(item, "title", None) or str(item)
        names.append(str(ref))
    return names
