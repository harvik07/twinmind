"""Kaggle credential resolution, value hygiene, the error taxonomy and secret safety.

These run without network access: resolution is driven by an injected environment
dict and temporary files. Only the live end-to-end check is marked ``data``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from twinmind.data import kaggle_auth
from twinmind.data.kaggle_auth import (
    ENV_API_TOKEN,
    ENV_KEY,
    ENV_USERNAME,
    KaggleAuthError,
    KaggleCredentialsNotFoundError,
    KaggleForbiddenError,
    KaggleUnauthorizedError,
    client_supports_access_token,
    looks_like_access_token,
    normalise_value,
    redact,
    resolve_credentials,
    status_code,
)
from twinmind.utils.paths import PROJECT_ROOT

FAKE_TOKEN = "KGAT_" + "a" * 32
FAKE_KEY = "0123456789abcdef0123456789abcdef"


@pytest.fixture()
def nowhere(tmp_path: Path) -> dict[str, Path]:
    """Paths that definitely hold no credentials."""
    return {
        "kaggle_json": tmp_path / "kaggle.json",
        "access_token_file": tmp_path / "access_token",
    }


# --------------------------------------------------------------------------- #
# Value hygiene
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("abc", "abc"),
        ("  abc  ", "abc"),
        ('"abc"', "abc"),
        ("'abc'", "abc"),
        ('  "abc"  ', "abc"),
        ('"  abc  "', "abc"),
        ("\ufeffabc", "abc"),
        ("", None),
        ("   ", None),
        (None, None),
    ],
)
def test_values_are_normalised(raw: str | None, expected: str | None) -> None:
    """Quotes, padding and a BOM are all invisible in a terminal and all cause 401s."""
    assert normalise_value(raw) == expected


def test_quoted_and_padded_credentials_resolve(nowhere: dict[str, Path]) -> None:
    env = {ENV_USERNAME: '  "someone"  ', ENV_KEY: f"'{FAKE_KEY}' "}
    resolved = resolve_credentials(env=env, **nowhere)

    assert resolved.username == "someone"
    assert resolved.key == FAKE_KEY


def test_looks_like_access_token() -> None:
    assert looks_like_access_token(FAKE_TOKEN)
    assert not looks_like_access_token(FAKE_KEY)
    assert not looks_like_access_token(None)


# --------------------------------------------------------------------------- #
# Redaction -- nothing may print a secret
# --------------------------------------------------------------------------- #
def test_redact_shows_only_a_prefix_and_length() -> None:
    out = redact(FAKE_TOKEN)
    assert out == f"KGAT...({len(FAKE_TOKEN)} chars)"
    assert FAKE_TOKEN not in out
    assert FAKE_TOKEN[4:] not in out


def test_redact_handles_empty() -> None:
    assert redact(None) == "<empty>"
    assert redact("") == "<empty>"


def test_describe_never_contains_the_secret() -> None:
    token = resolve_credentials(
        env={ENV_API_TOKEN: FAKE_TOKEN},
        kaggle_json=Path("/nonexistent"),
        access_token_file=Path("/nonexistent"),
    )
    assert FAKE_TOKEN not in token.describe()

    legacy = resolve_credentials(
        env={ENV_USERNAME: "someone", ENV_KEY: FAKE_KEY},
        kaggle_json=Path("/nonexistent"),
        access_token_file=Path("/nonexistent"),
    )
    assert FAKE_KEY not in legacy.describe()


def test_no_secret_reaches_the_logs(
    caplog: pytest.LogCaptureFixture, nowhere: dict[str, Path]
) -> None:
    """The whole point of the module: resolution must never log a credential."""
    caplog.set_level(logging.DEBUG)
    logger = logging.getLogger("twinmind.data.kaggle_auth")
    logger.propagate = True  # get_logger detaches it from the root by default
    try:
        # The misplaced-token path is the chattiest, so exercise that one.
        resolve_credentials(env={ENV_USERNAME: "someone", ENV_KEY: FAKE_TOKEN}, **nowhere)
    finally:
        logger.propagate = False

    blob = caplog.text
    assert blob, "expected the resolver to log something"
    assert FAKE_TOKEN not in blob
    assert FAKE_TOKEN[5:] not in blob, "the token body leaked into the log"
    assert "KGAT..." in blob, "the redacted form should still be present"


# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not client_supports_access_token(),
    reason="installed kaggle client has no access-token support",
)
def test_api_token_wins_over_legacy_pair(nowhere: dict[str, Path]) -> None:
    resolved = resolve_credentials(
        env={ENV_API_TOKEN: FAKE_TOKEN, ENV_USERNAME: "someone", ENV_KEY: FAKE_KEY},
        **nowhere,
    )
    assert resolved.scheme == "access_token"
    assert resolved.api_token == FAKE_TOKEN
    assert ENV_API_TOKEN in resolved.source


def test_environment_pair_wins_over_kaggle_json(tmp_path: Path) -> None:
    kaggle_json = tmp_path / "kaggle.json"
    kaggle_json.write_text(json.dumps({"username": "from-file", "key": "f" * 32}), encoding="utf-8")

    resolved = resolve_credentials(
        env={ENV_USERNAME: "from-env", ENV_KEY: FAKE_KEY},
        kaggle_json=kaggle_json,
        access_token_file=tmp_path / "absent",
    )
    assert resolved.username == "from-env"
    assert resolved.source == "environment"


def test_kaggle_json_is_the_last_resort(tmp_path: Path) -> None:
    kaggle_json = tmp_path / "kaggle.json"
    kaggle_json.write_text(json.dumps({"username": "from-file", "key": FAKE_KEY}), encoding="utf-8")

    resolved = resolve_credentials(
        env={}, kaggle_json=kaggle_json, access_token_file=tmp_path / "absent"
    )
    assert resolved.scheme == "legacy_api_key"
    assert resolved.username == "from-file"
    assert str(kaggle_json) in resolved.source


def test_kaggle_json_tolerates_a_bom(tmp_path: Path) -> None:
    kaggle_json = tmp_path / "kaggle.json"
    kaggle_json.write_bytes(
        b"\xef\xbb\xbf" + json.dumps({"username": "u", "key": FAKE_KEY}).encode("utf-8")
    )
    resolved = resolve_credentials(
        env={}, kaggle_json=kaggle_json, access_token_file=tmp_path / "absent"
    )
    assert resolved.username == "u"


@pytest.mark.skipif(
    not client_supports_access_token(),
    reason="installed kaggle client has no access-token support",
)
def test_a_kgat_token_in_kaggle_key_is_routed_as_a_token(nowhere: dict[str, Path]) -> None:
    """The exact misconfiguration that produced the original 401.

    A KGAT_ value in KAGGLE_KEY is an access token, not a legacy key. Sent as a
    Basic-auth password it is rejected; recognised for what it is, it works.
    """
    resolved = resolve_credentials(env={ENV_USERNAME: "someone", ENV_KEY: FAKE_TOKEN}, **nowhere)
    assert resolved.scheme == "access_token"
    assert resolved.api_token == FAKE_TOKEN


@pytest.mark.skipif(
    not client_supports_access_token(),
    reason="installed kaggle client has no access-token support",
)
def test_access_token_file_is_read(tmp_path: Path) -> None:
    token_file = tmp_path / "access_token"
    token_file.write_text(f"  {FAKE_TOKEN}\n", encoding="utf-8")
    resolved = resolve_credentials(
        env={}, kaggle_json=tmp_path / "absent", access_token_file=token_file
    )
    assert resolved.scheme == "access_token"
    assert resolved.api_token == FAKE_TOKEN


def test_as_environment_presents_only_the_resolved_scheme(nowhere: dict[str, Path]) -> None:
    legacy = resolve_credentials(env={ENV_USERNAME: "u", ENV_KEY: FAKE_KEY}, **nowhere)
    assert set(legacy.as_environment()) == {ENV_USERNAME, ENV_KEY}


# --------------------------------------------------------------------------- #
# Error taxonomy -- three distinct, actionable messages
# --------------------------------------------------------------------------- #
def test_no_credentials_names_every_location_checked(nowhere: dict[str, Path]) -> None:
    with pytest.raises(KaggleCredentialsNotFoundError) as excinfo:
        resolve_credentials(env={}, **nowhere)

    message = str(excinfo.value)
    assert "NO KAGGLE CREDENTIALS FOUND" in message
    assert ENV_API_TOKEN in message
    assert ENV_USERNAME in message and ENV_KEY in message
    assert "kaggle.json" in message
    assert "settings" in message


def test_unauthorized_names_the_source_and_says_regenerate() -> None:
    message = str(KaggleUnauthorizedError("$KAGGLE_KEY (access token)"))
    assert "401" in message
    assert "$KAGGLE_KEY (access token)" in message
    assert "kaggle.com/settings" in message
    assert "403" not in message


def test_forbidden_gives_the_competition_url_and_rules_wording() -> None:
    message = str(KaggleForbiddenError("store-sales-time-series-forecasting", "environment"))
    assert "403" in message
    assert "rules have not been accepted" in message
    assert "kaggle.com/competitions/store-sales-time-series-forecasting" in message
    assert "401" not in message


def test_the_three_errors_are_distinguishable() -> None:
    messages = {
        str(KaggleCredentialsNotFoundError(["a"])),
        str(KaggleUnauthorizedError("src")),
        str(KaggleForbiddenError("slug", "src")),
    }
    assert len(messages) == 3
    for message in messages:
        assert issubclass(KaggleAuthError, RuntimeError)
        assert "Traceback" not in message


@pytest.mark.parametrize(
    "exc,expected",
    [
        (type("E", (Exception,), {"status": 401})(), 401),
        (type("E", (Exception,), {"code": 403})(), 403),
        (Exception("HTTP 401 Unauthorized"), 401),
        (Exception("boom"), None),
    ],
)
def test_status_code_extraction(exc: BaseException, expected: int | None) -> None:
    assert status_code(exc) == expected


# --------------------------------------------------------------------------- #
# CLI behaviour -- a missing credential must exit non-zero with the message
# --------------------------------------------------------------------------- #
def test_missing_credentials_exit_non_zero_without_a_traceback(tmp_path: Path) -> None:
    """Runs the real CLI with the credential environment stripped."""
    # Inherit the real environment minus every credential: a hand-built minimal
    # env loses PATH and the interpreter cannot then import its own dependencies.
    env = {k: v for k, v in os.environ.items() if not k.startswith("KAGGLE_")}
    env.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "HOME": str(tmp_path),
            "USERPROFILE": str(tmp_path),
            "KAGGLE_CONFIG_DIR": str(tmp_path / "no-kaggle"),
            "TWINMIND_SKIP_DOTENV": "1",
        }
    )
    # --check-auth always authenticates; --dataset favorita would short-circuit
    # on a satisfied manifest once the data is already downloaded.
    result = subprocess.run(
        [sys.executable, "scripts/download_data.py", "--check-auth"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "NO KAGGLE CREDENTIALS FOUND" in combined
    assert "Traceback" not in combined


# --------------------------------------------------------------------------- #
# Live check
# --------------------------------------------------------------------------- #
@pytest.mark.data
def test_check_auth_against_the_real_api() -> None:
    info = kaggle_auth.check_auth("store-sales-time-series-forecasting")
    assert info["username"]
    assert info["auth_method"]
    assert info["client_version"] != "unknown"
