"""LLM access for the agent layer.

The LLM does exactly two things in this project: it turns a natural-language
goal into an ``ObjectiveSpec`` (planner) and it writes prose from a structured
facts object (explainer). It never chooses a plan number. Every KPI, comparison
and constraint check is deterministic Python elsewhere.

Pieces:

* :class:`LLMClient` -- the protocol every backend satisfies: one
  ``complete(messages, json_mode)`` round trip returning an :class:`LLMReply`.
* :class:`GroqClient` -- the real backend: timeouts, retry with exponential
  backoff on transient errors, a token-bucket rate limiter, a JSONL call log.
* :class:`FakeLLM` -- scripted replies, so every graph path is testable offline.
* :class:`CallBudget` -- a hard cap on LLM calls per goal, wrapping any client.
  Every round trip, including spec repairs and explanation regenerations, goes
  through it, so the cap cannot be bypassed by an inner retry loop.
* :func:`discover_model` -- picks the model from Groq's live model list by the
  rule in ``configs/agent.yaml``; no model id is hard-coded.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

Message = dict[str, str]

#: Rough characters-per-token ratio, used only to pre-charge the token bucket.
_CHARS_PER_TOKEN = 4.0


class LLMError(RuntimeError):
    """A backend could not produce a response."""


class LLMBudgetError(LLMError):
    """The per-goal LLM call cap would be exceeded."""


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
class TokenBucket:
    """A classic token bucket with an injectable clock (so tests never sleep)."""

    def __init__(
        self,
        capacity: float,
        refill_per_second: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0 or refill_per_second <= 0:
            raise ValueError("capacity and refill_per_second must be positive")
        self.capacity = float(capacity)
        self.refill_per_second = float(refill_per_second)
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_second)

    @property
    def available(self) -> float:
        """Tokens currently in the bucket."""
        self._refill()
        return self._tokens

    def wait_time(self, amount: float) -> float:
        """Seconds until ``amount`` tokens would be available. 0.0 if available now."""
        if amount > self.capacity:
            raise ValueError(
                f"requested {amount} exceeds bucket capacity {self.capacity}; "
                "the request can never be admitted"
            )
        self._refill()
        if self._tokens >= amount:
            return 0.0
        return (amount - self._tokens) / self.refill_per_second

    def consume(self, amount: float) -> None:
        """Deduct ``amount`` tokens without checking. Call after ``wait_time``."""
        self._refill()
        self._tokens -= amount


class RateLimiter:
    """Requests-per-minute and tokens-per-minute limits, enforced together.

    Both the clock and the sleep function are injectable: tests advance a fake
    clock and assert on the *requested* wait instead of actually blocking.
    """

    def __init__(
        self,
        requests_per_minute: int = 28,
        tokens_per_minute: int = 5500,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._clock = clock
        self._sleeper = sleeper
        self._requests = TokenBucket(requests_per_minute, requests_per_minute / 60.0, clock)
        self._tokens = TokenBucket(tokens_per_minute, tokens_per_minute / 60.0, clock)

    def wait_time(self, estimated_tokens: int) -> float:
        """Seconds this call would have to wait before it may proceed."""
        return max(self._requests.wait_time(1.0), self._tokens.wait_time(float(estimated_tokens)))

    def acquire(self, estimated_tokens: int) -> float:
        """Block until the call may proceed. Returns the seconds waited."""
        waited = self.wait_time(estimated_tokens)
        if waited > 0:
            logger.debug("rate limiter sleeping %.2fs", waited)
            self._sleeper(waited)
        self._requests.consume(1.0)
        self._tokens.consume(float(estimated_tokens))
        return waited

    def settle(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Correct the token bucket once real usage is known."""
        self._tokens.consume(float(actual_tokens - estimated_tokens))


# --------------------------------------------------------------------------- #
# Call logging
# --------------------------------------------------------------------------- #
class CallLogger:
    """Appends one JSON object per LLM call. Never logs message content or keys."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **record: Any) -> None:
        """Append a record. Logging must never break an agent run, so errors are swallowed."""
        record.setdefault("ts", time.time())
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # pragma: no cover - disk errors are not worth failing a run
            logger.warning("could not write LLM call log: %s", exc)


def estimate_tokens(messages: Sequence[Message], max_tokens: int) -> int:
    """Cheap pre-flight token estimate: prompt characters / 4, plus the output budget."""
    chars = sum(len(m.get("content", "")) for m in messages)
    return int(chars / _CHARS_PER_TOKEN) + max_tokens


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LLMReply:
    """One completed round trip."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    transient_retries: int = 0

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    """What the agent may call. One round trip per ``complete``; no hidden retries
    on content -- a reply that fails validation is the CALLER's to repair, so
    every repair is visible to the per-goal call budget."""

    model: str

    def complete(self, messages: Sequence[Message], json_mode: bool = False) -> LLMReply:
        """Send ``messages`` and return the reply."""
        ...


# --------------------------------------------------------------------------- #
# Groq
# --------------------------------------------------------------------------- #
#: Errors worth a retry: the request did not complete, so nothing was spent.
_TRANSIENT = (
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
)


def load_api_key(env_name: str, env_file: Path | None = None) -> str:
    """The key from the environment or ``.env``. Never logged or printed."""
    key = os.getenv(env_name)
    if not key and env_file is not None and env_file.exists():
        from dotenv import dotenv_values

        key = dotenv_values(env_file).get(env_name) or None
    if not key:
        raise LLMError(f"{env_name} is not set in the environment or in .env.")
    return key


class GroqClient:
    """Groq chat completions with timeouts, backoff, rate limiting and a call log."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_seconds: float = 30.0,
        max_transient_retries: int = 3,
        backoff_seconds: float = 2.0,
        extra_params: dict[str, Any] | None = None,
        client: Any | None = None,
        rate_limiter: RateLimiter | None = None,
        call_logger: CallLogger | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_transient_retries = max_transient_retries
        self.backoff_seconds = backoff_seconds
        self.extra_params = dict(extra_params or {})
        self.rate_limiter = rate_limiter or RateLimiter()
        self.call_logger = call_logger
        self._sleep = sleeper
        if client is not None:
            self._client = client
        else:
            if not api_key:
                raise LLMError("GroqClient needs an API key.")
            from groq import Groq

            # max_retries=0: retries are ours, so each one is counted and logged.
            self._client = Groq(api_key=api_key, timeout=timeout_seconds, max_retries=0)

    def complete(self, messages: Sequence[Message], json_mode: bool = False) -> LLMReply:
        """One logical call, with up to ``max_transient_retries`` transport retries."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_params,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        estimated = estimate_tokens(messages, self.max_tokens)
        retries = 0
        while True:
            self.rate_limiter.acquire(estimated)
            started = time.monotonic()
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as error:  # noqa: BLE001 - classified below
                if type(error).__name__ in _TRANSIENT and retries < self.max_transient_retries:
                    delay = self.backoff_seconds * (2**retries)
                    retries += 1
                    logger.warning(
                        "groq %s; retry %d/%d in %.1fs",
                        type(error).__name__,
                        retries,
                        self.max_transient_retries,
                        delay,
                    )
                    self._sleep(delay)
                    continue
                raise LLMError(f"groq call failed: {type(error).__name__}: {error}") from error
            latency = time.monotonic() - started
            break
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        if prompt_tokens + completion_tokens:
            self.rate_limiter.settle(estimated, prompt_tokens + completion_tokens)
        reply = LLMReply(
            text=response.choices[0].message.content or "",
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency,
            transient_retries=retries,
        )
        if self.call_logger is not None:
            self.call_logger.log(
                backend="groq",
                model=self.model,
                json_mode=json_mode,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_seconds=round(latency, 4),
                transient_retries=retries,
            )
        return reply


# --------------------------------------------------------------------------- #
# Fake
# --------------------------------------------------------------------------- #
Script = str | Callable[[Sequence[Message]], str]


@dataclass
class FakeLLM:
    """Scripted replies, in order; each may be a string or a function of the messages.

    Records every request so a test can assert on what the model was shown.
    Running out of script is an error rather than a silent default, so a test
    cannot pass by accident on an unscripted call.
    """

    script: list[Script]
    model: str = "fake-llm"
    calls: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: Sequence[Message], json_mode: bool = False) -> LLMReply:
        """Return the next scripted reply."""
        self.calls.append(list(messages))
        if len(self.calls) > len(self.script):
            raise LLMError(f"FakeLLM ran out of script after {len(self.script)} replies")
        item = self.script[len(self.calls) - 1]
        text = item(messages) if callable(item) else item
        chars = sum(len(m.get("content", "")) for m in messages)
        return LLMReply(
            text=text,
            model=self.model,
            prompt_tokens=int(chars / _CHARS_PER_TOKEN),
            completion_tokens=int(len(text) / _CHARS_PER_TOKEN),
        )


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
@dataclass
class CallRecord:
    """One LLM round trip, for the trace."""

    node: str
    purpose: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    transient_retries: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return dict(self.__dict__)


class CallBudget:
    """Wraps a client and refuses the call that would exceed ``max_calls``."""

    def __init__(self, client: LLMClient, max_calls: int) -> None:
        self.client = client
        self.max_calls = max_calls
        self.records: list[CallRecord] = []

    @property
    def used(self) -> int:
        """Calls made so far."""
        return len(self.records)

    @property
    def remaining(self) -> int:
        """Calls still allowed."""
        return self.max_calls - self.used

    def complete(
        self, messages: Sequence[Message], *, node: str, purpose: str, json_mode: bool = False
    ) -> LLMReply:
        """One counted call, or :class:`LLMBudgetError` before any is made."""
        if self.used >= self.max_calls:
            raise LLMBudgetError(
                f"LLM call cap of {self.max_calls} per goal reached ({node}/{purpose})"
            )
        reply = self.client.complete(messages, json_mode=json_mode)
        self.records.append(
            CallRecord(
                node=node,
                purpose=purpose,
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
                latency_seconds=reply.latency_seconds,
                transient_retries=reply.transient_retries,
            )
        )
        return reply


# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelRule:
    """How to pick a model from Groq's live list (``configs/agent.yaml``)."""

    exclude_patterns: tuple[str, ...]
    min_context_window: int
    min_completion_tokens: int


def extra_params_for(model_id: str, by_prefix: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Model-family request parameters (e.g. reasoning effort), longest prefix wins."""
    params: dict[str, Any] = {}
    for prefix in sorted(by_prefix, key=len):
        if model_id.startswith(prefix):
            params = dict(by_prefix[prefix])
    return params


def _size_billions(model_id: str) -> float:
    """Parameter count from the id (``...-20b`` -> 20); unknown sorts last."""
    match = re.search(r"(\d+(?:\.\d+)?)b\b", model_id.lower().split("/")[-1])
    return float(match.group(1)) if match else float("inf")


def rank_models(models: Sequence[Any], rule: ModelRule) -> list[str]:
    """Eligible model ids, fastest (smallest) first, ties by id.

    "Fast" is read as "fewest parameters": on the same serving stack, a smaller
    model returns sooner and draws less of the free-tier token budget.
    """
    eligible = []
    for model in models:
        model_id = str(getattr(model, "id", ""))
        if not model_id or getattr(model, "active", True) is False:
            continue
        if any(p in model_id.lower() for p in rule.exclude_patterns):
            continue
        if int(getattr(model, "context_window", 0) or 0) < rule.min_context_window:
            continue
        if int(getattr(model, "max_completion_tokens", 0) or 0) < rule.min_completion_tokens:
            continue
        eligible.append(model_id)
    return sorted(eligible, key=lambda m: (_size_billions(m), m))


def discover_model(raw_client: Any, rule: ModelRule, probe: Callable[[str], bool]) -> str:
    """The first ranked model that passes a JSON-mode probe.

    ``probe(model_id)`` makes one tiny JSON-mode call and returns whether it
    parsed. At most one probe per candidate, stopping at the first success.
    """
    ranked = rank_models(raw_client.models.list().data, rule)
    if not ranked:
        raise LLMError("no Groq model matches the selection rule in configs/agent.yaml")
    for model_id in ranked:
        if probe(model_id):
            return model_id
        logger.warning("model %s failed the JSON-mode probe; trying the next", model_id)
    raise LLMError(f"no candidate passed the JSON-mode probe: {ranked}")
