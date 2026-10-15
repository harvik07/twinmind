"""The LLM client layer, offline: transport retries, the call cap, discovery, rate limits.

Nothing here touches the network. GroqClient runs against a fake transport.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from twinmind.agents.llm_client import (
    CallBudget,
    CallLogger,
    FakeLLM,
    GroqClient,
    LLMBudgetError,
    LLMError,
    ModelRule,
    RateLimiter,
    TokenBucket,
    discover_model,
    estimate_tokens,
    extra_params_for,
    rank_models,
)


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


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
    limiter = RateLimiter(28, 5500, clock=clock, sleeper=sleeps.append)
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
class RateLimitError(Exception):
    """Named like groq's, which is how the client classifies transient errors."""


class BadRequestError(Exception):
    """A non-transient error: retrying would not help."""


def _response(content: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class _Transport:
    """Fake ``client.chat.completions``: raises the queued errors, then answers."""

    def __init__(self, content: str, errors: list[Exception] | None = None) -> None:
        self.content = content
        self.errors = list(errors or [])
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return _response(self.content)


def _client(transport: _Transport, tmp_path: Path, **kwargs: Any) -> tuple[GroqClient, list[float]]:
    sleeps: list[float] = []
    client = GroqClient(
        "some/model-8b",
        client=transport,
        rate_limiter=RateLimiter(28, 5500, clock=FakeClock(), sleeper=lambda _: None),
        call_logger=CallLogger(tmp_path / "calls.jsonl"),
        sleeper=sleeps.append,
        **kwargs,
    )
    return client, sleeps


def test_groq_client_returns_text_and_usage(tmp_path: Path) -> None:
    client, _ = _client(_Transport('{"ok": true}'), tmp_path)
    reply = client.complete([{"role": "user", "content": "hi"}], json_mode=True)
    assert reply.text == '{"ok": true}'
    assert (reply.prompt_tokens, reply.completion_tokens, reply.total_tokens) == (10, 5, 15)


def test_groq_client_requests_json_mode_only_when_asked(tmp_path: Path) -> None:
    transport = _Transport("x")
    client, _ = _client(transport, tmp_path, extra_params={"reasoning_effort": "low"})
    client.complete([{"role": "user", "content": "a"}], json_mode=True)
    client.complete([{"role": "user", "content": "b"}])
    assert transport.calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in transport.calls[1]
    assert transport.calls[0]["reasoning_effort"] == "low"


def test_transient_errors_are_retried_with_exponential_backoff(tmp_path: Path) -> None:
    transport = _Transport("done", errors=[RateLimitError("429"), RateLimitError("429")])
    client, sleeps = _client(transport, tmp_path, backoff_seconds=2.0, max_transient_retries=3)
    reply = client.complete([{"role": "user", "content": "x"}])
    assert reply.text == "done"
    assert reply.transient_retries == 2
    assert sleeps == [2.0, 4.0]


def test_transient_retries_are_bounded(tmp_path: Path) -> None:
    transport = _Transport("never", errors=[RateLimitError("429")] * 5)
    client, sleeps = _client(transport, tmp_path, max_transient_retries=2)
    with pytest.raises(LLMError, match="RateLimitError"):
        client.complete([{"role": "user", "content": "x"}])
    assert len(sleeps) == 2 and len(transport.calls) == 3


def test_a_non_transient_error_is_not_retried(tmp_path: Path) -> None:
    transport = _Transport("never", errors=[BadRequestError("400")])
    client, sleeps = _client(transport, tmp_path)
    with pytest.raises(LLMError, match="BadRequestError"):
        client.complete([{"role": "user", "content": "x"}])
    assert sleeps == [] and len(transport.calls) == 1


def test_the_call_log_records_usage_but_never_content(tmp_path: Path) -> None:
    client, _ = _client(_Transport("secret answer"), tmp_path)
    client.complete([{"role": "user", "content": "secret question"}])
    log = (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
    record = json.loads(log.strip())
    assert record["prompt_tokens"] == 10
    assert "secret" not in log


# --------------------------------------------------------------------------- #
# FakeLLM and the per-goal cap
# --------------------------------------------------------------------------- #
def test_fake_llm_serves_its_script_and_then_refuses() -> None:
    fake = FakeLLM(["one", lambda messages: f"saw {len(messages)}"])
    assert fake.complete([{"role": "user", "content": "a"}]).text == "one"
    assert fake.complete([{"role": "user", "content": "b"}]).text == "saw 1"
    with pytest.raises(LLMError, match="ran out of script"):
        fake.complete([{"role": "user", "content": "c"}])


def test_call_budget_refuses_the_call_beyond_the_cap() -> None:
    fake = FakeLLM(["r"] * 10)
    budget = CallBudget(fake, max_calls=6)
    for _ in range(6):
        budget.complete([{"role": "user", "content": "x"}], node="n", purpose="p")
    with pytest.raises(LLMBudgetError):
        budget.complete([{"role": "user", "content": "x"}], node="n", purpose="p")
    assert budget.used == 6
    assert len(fake.calls) == 6, "the refused call must not reach the model"


# --------------------------------------------------------------------------- #
# Model discovery -- the rule picks from the live list; nothing hard-coded
# --------------------------------------------------------------------------- #
def _model(model_id: str, ctx: int = 131072, out: int = 32768, active: bool = True) -> Any:
    return SimpleNamespace(
        id=model_id, context_window=ctx, max_completion_tokens=out, active=active
    )


RULE = ModelRule(
    exclude_patterns=("whisper", "guard", "tts"),
    min_context_window=32768,
    min_completion_tokens=4096,
)


def test_rank_models_filters_by_the_rule_and_orders_smallest_first() -> None:
    models = [
        _model("vendor/big-120b"),
        _model("vendor/small-20b"),
        _model("whisper-large-v3"),
        _model("meta/llama-prompt-guard-2-86m"),
        _model("tiny-7b", ctx=4096),
        _model("retired-8b", active=False),
        _model("vendor/mid-27b"),
    ]
    assert rank_models(models, RULE) == ["vendor/small-20b", "vendor/mid-27b", "vendor/big-120b"]


def test_discover_model_takes_the_first_candidate_that_passes_the_probe() -> None:
    listing = SimpleNamespace(data=[_model("a/m-20b"), _model("b/m-70b")])
    raw = SimpleNamespace(models=SimpleNamespace(list=lambda: listing))
    probed: list[str] = []

    def probe(model_id: str) -> bool:
        probed.append(model_id)
        return model_id == "b/m-70b"

    assert discover_model(raw, RULE, probe) == "b/m-70b"
    assert probed == ["a/m-20b", "b/m-70b"]


def test_extra_params_follow_the_model_family() -> None:
    by_prefix = {"openai/gpt-oss": {"reasoning_effort": "low"}}
    assert extra_params_for("openai/gpt-oss-20b", by_prefix) == {"reasoning_effort": "low"}
    assert extra_params_for("qwen/qwen3-32b", by_prefix) == {}
