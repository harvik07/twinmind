"""EchoClient must be offline and schema-valid; the rate limiter must actually bite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, ValidationError

from twinmind.agents.llm_client import (
    CallLogger,
    EchoClient,
    GroqClient,
    LLMError,
    RateLimiter,
    TokenBucket,
    estimate_tokens,
    get_llm_client,
)
from twinmind.utils.config import LLMConfig


class Proposal(BaseModel):
    """A schema of the shape the Explorer node will actually ask for."""

    rationale: str
    shift_count: int = Field(ge=1, le=3)
    overtime_hours: float = Field(ge=0.0)


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --------------------------------------------------------------------------- #
# EchoClient
# --------------------------------------------------------------------------- #
def test_echo_client_returns_a_schema_valid_object(llm_config: LLMConfig) -> None:
    client = EchoClient(llm_config)
    result = client.complete([{"role": "user", "content": "propose a scenario"}], schema=Proposal)

    assert isinstance(result, Proposal)
    Proposal.model_validate(result.model_dump())  # round-trips cleanly
    assert client.call_count == 1


def test_echo_client_without_schema_returns_text(llm_config: LLMConfig) -> None:
    client = EchoClient(llm_config)
    result = client.complete([{"role": "user", "content": "explain"}])
    assert isinstance(result, str)
    assert "explain" in result


def test_echo_client_is_deterministic(llm_config: LLMConfig) -> None:
    messages = [{"role": "user", "content": "same input"}]
    first = EchoClient(llm_config).complete(messages, schema=Proposal)
    second = EchoClient(llm_config).complete(messages, schema=Proposal)
    assert isinstance(first, Proposal) and isinstance(second, Proposal)
    assert first.model_dump() == second.model_dump()


def test_echo_client_serves_canned_responses_in_order(llm_config: LLMConfig) -> None:
    canned = [Proposal(rationale="first", shift_count=3, overtime_hours=2.0), "plain text"]
    client = EchoClient(llm_config, responses=canned)
    assert client.complete([], schema=Proposal) == canned[0]
    assert client.complete([]) == "plain text"


def test_echo_client_writes_the_call_log(llm_config: LLMConfig, tmp_path: Path) -> None:
    logger = CallLogger(tmp_path / "calls.jsonl")
    client = EchoClient(llm_config, call_logger=logger)
    client.complete([{"role": "user", "content": "hello"}], schema=Proposal)

    lines = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["backend"] == "echo"
    assert record["schema"] == "Proposal"
    assert "latency_seconds" in record and "total_tokens" in record


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_factory_returns_echo_when_configured(llm_config: LLMConfig) -> None:
    assert isinstance(get_llm_client(llm_config), EchoClient)


def test_factory_groq_without_a_key_fails_clearly(
    llm_config: LLMConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    groq_config = llm_config.model_copy(update={"provider": "groq"})
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        get_llm_client(groq_config)


# --------------------------------------------------------------------------- #
# Rate limiting -- fake clock, never sleeps
# --------------------------------------------------------------------------- #
def test_token_bucket_refills_over_time() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=10, refill_per_second=1.0, clock=clock)
    bucket.consume(10)
    assert bucket.wait_time(5) == pytest.approx(5.0)
    clock.advance(5.0)
    assert bucket.wait_time(5) == 0.0


def test_rate_limiter_blocks_the_twenty_ninth_request_within_a_minute() -> None:
    """28 RPM: calls 1-28 pass instantly, call 29 must wait."""
    clock = FakeClock()
    sleeps: list[float] = []
    limiter = RateLimiter(
        requests_per_minute=28,
        tokens_per_minute=5500,
        clock=clock,
        sleeper=sleeps.append,
    )

    for i in range(28):
        assert limiter.acquire(estimated_tokens=1) == 0.0, f"call {i + 1} should not have waited"

    assert sleeps == []

    waited = limiter.acquire(estimated_tokens=1)
    assert waited > 0.0, "the 29th call within the minute must be throttled"
    assert sleeps == [waited]
    assert waited == pytest.approx(60.0 / 28.0)


def test_rate_limiter_also_throttles_on_tokens() -> None:
    clock = FakeClock()
    sleeps: list[float] = []
    limiter = RateLimiter(
        requests_per_minute=28,
        tokens_per_minute=5500,
        clock=clock,
        sleeper=sleeps.append,
    )
    assert limiter.acquire(estimated_tokens=5500) == 0.0
    waited = limiter.acquire(estimated_tokens=1000)
    assert waited > 0.0
    assert sleeps == [waited]


def test_rate_limiter_recovers_after_the_window_passes() -> None:
    clock = FakeClock()
    limiter = RateLimiter(28, 5500, clock=clock, sleeper=lambda _: None)
    for _ in range(28):
        limiter.acquire(1)
    clock.advance(60.0)
    assert limiter.acquire(1) == 0.0


def test_request_larger_than_the_bucket_is_rejected() -> None:
    limiter = RateLimiter(28, 5500, clock=FakeClock(), sleeper=lambda _: None)
    with pytest.raises(ValueError, match="never be admitted"):
        limiter.acquire(estimated_tokens=6000)


def test_estimate_tokens_counts_prompt_and_output_budget() -> None:
    messages = [{"role": "user", "content": "x" * 400}]
    assert estimate_tokens(messages, max_tokens=100) == 200


# --------------------------------------------------------------------------- #
# GroqClient against a fake transport (still no network)
# --------------------------------------------------------------------------- #
class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self, contents: list[str]) -> None:
        self._contents = contents
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._contents) - 1)
        return _FakeResponse(self._contents[index])


class _FakeGroq:
    def __init__(self, contents: list[str]) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions(contents)})()


def _groq_client(contents: list[str], config: LLMConfig, tmp_path: Path) -> GroqClient:
    return GroqClient(
        config.model_copy(update={"provider": "groq"}),
        client=_FakeGroq(contents),
        rate_limiter=RateLimiter(28, 5500, clock=FakeClock(), sleeper=lambda _: None),
        call_logger=CallLogger(tmp_path / "calls.jsonl"),
    )


def test_groq_client_parses_a_valid_json_response(llm_config: LLMConfig, tmp_path: Path) -> None:
    body = json.dumps({"rationale": "run three shifts", "shift_count": 3, "overtime_hours": 1.0})
    client = _groq_client([body], llm_config, tmp_path)
    result = client.complete([{"role": "user", "content": "propose"}], schema=Proposal)
    assert isinstance(result, Proposal)
    assert result.shift_count == 3


def test_groq_client_retries_then_succeeds(llm_config: LLMConfig, tmp_path: Path) -> None:
    good = json.dumps({"rationale": "ok", "shift_count": 2, "overtime_hours": 0.0})
    client = _groq_client(["not json at all", good], llm_config, tmp_path)
    result = client.complete([{"role": "user", "content": "propose"}], schema=Proposal)
    assert isinstance(result, Proposal)

    records = [
        json.loads(line)
        for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert [r["attempt"] for r in records] == [1, 2]
    assert all(r["total_tokens"] == 15 for r in records)


def test_groq_client_gives_up_after_the_configured_retries(
    llm_config: LLMConfig, tmp_path: Path
) -> None:
    config = llm_config.model_copy(update={"provider": "groq", "max_parse_retries": 2})
    client = GroqClient(
        config,
        client=_FakeGroq(["nonsense"]),
        rate_limiter=RateLimiter(28, 5500, clock=FakeClock(), sleeper=lambda _: None),
        call_logger=CallLogger(tmp_path / "calls.jsonl"),
    )
    with pytest.raises(LLMError, match="3 attempts"):
        client.complete([{"role": "user", "content": "propose"}], schema=Proposal)


def test_groq_client_rejects_a_schema_violating_response(
    llm_config: LLMConfig, tmp_path: Path
) -> None:
    """Valid JSON, invalid against the schema -- shift_count is out of range."""
    bad = json.dumps({"rationale": "too many", "shift_count": 9, "overtime_hours": 0.0})
    client = _groq_client([bad], llm_config, tmp_path)
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "propose"}], schema=Proposal)
    with pytest.raises(ValidationError):
        Proposal.model_validate_json(bad)


def test_groq_client_requests_json_mode(llm_config: LLMConfig, tmp_path: Path) -> None:
    body = json.dumps({"rationale": "x", "shift_count": 1, "overtime_hours": 0.0})
    client = _groq_client([body], llm_config, tmp_path)
    client.complete([{"role": "user", "content": "propose"}], schema=Proposal)
    sent = client._client.chat.completions.calls[0]
    assert sent["response_format"] == {"type": "json_object"}


def test_groq_client_without_schema_returns_raw_text(llm_config: LLMConfig, tmp_path: Path) -> None:
    client = _groq_client(["just prose"], llm_config, tmp_path)
    assert client.complete([{"role": "user", "content": "explain"}]) == "just prose"
