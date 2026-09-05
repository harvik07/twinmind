"""LLM access for the agent layer.

The agent may only use an LLM to *propose* scenario parameters and to *write*
explanations. Every KPI, comparison and constraint check is deterministic Python
elsewhere in the codebase; nothing here is allowed to decide a number that ends
up in a result table.

Three pieces:

* ``LLMClient`` - the abstract interface every backend implements.
* ``GroqClient`` - the real backend: JSON mode, schema validation with retries,
  a token-bucket rate limiter, and a JSONL call log for the calls-per-query metric.
* ``EchoClient`` - a deterministic, offline stub used by tests.

``get_llm_client()`` chooses between them from ``configs/agent.yaml``, so adding
an Ollama backend later means adding a class here and one line in the factory.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from twinmind.utils.config import AgentConfig, LLMConfig, load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger(__name__)

Message = dict[str, str]
T = TypeVar("T", bound=BaseModel)

#: Rough characters-per-token ratio, used only to pre-charge the token bucket.
_CHARS_PER_TOKEN = 4.0


class LLMError(RuntimeError):
    """Raised when a backend cannot produce a usable response."""


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
    """Appends one JSON object per LLM call, for the calls-per-query metric."""

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


def _schema_instruction(schema: type[BaseModel]) -> str:
    """The system-message fragment that pins the model to a JSON schema."""
    return (
        "Respond with a single JSON object and nothing else. It must validate "
        f"against this JSON Schema:\n{json.dumps(schema.model_json_schema(), indent=2)}"
    )


def _extract_json(text: str) -> str:
    """Pull the outermost JSON object out of a response that may be fenced."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end <= start:
        raise LLMError(f"no JSON object found in response: {text[:200]!r}")
    return stripped[start : end + 1]


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class LLMClient(ABC):
    """Backend-agnostic LLM interface."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        schema: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> BaseModel | str:
        """Return a validated ``schema`` instance, or raw text when ``schema`` is None."""

    def complete_as(
        self,
        messages: Sequence[Message],
        schema: type[T],
        model: str | None = None,
    ) -> T:
        """Typed convenience wrapper around ``complete`` for a known schema."""
        return cast(T, self.complete(messages, schema=schema, model=model))


# --------------------------------------------------------------------------- #
# Groq
# --------------------------------------------------------------------------- #
class GroqClient(LLMClient):
    """Groq backend with JSON mode, schema retries, rate limiting and call logging."""

    def __init__(
        self,
        config: LLMConfig,
        api_key: str | None = None,
        client: Any | None = None,
        rate_limiter: RateLimiter | None = None,
        call_logger: CallLogger | None = None,
    ) -> None:
        super().__init__(config)
        self.rate_limiter = rate_limiter or RateLimiter(
            requests_per_minute=config.rate_limit.requests_per_minute,
            tokens_per_minute=config.rate_limit.tokens_per_minute,
        )
        self.call_logger = call_logger or CallLogger(PROJECT_ROOT / config.call_log)
        if client is not None:
            self._client = client
            return
        key = api_key or os.getenv(config.api_key_env)
        if not key:
            raise LLMError(
                f"{config.api_key_env} is not set. Copy .env.example to .env and fill it in."
            )
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - groq is a hard dependency
            raise LLMError("the 'groq' package is not installed") from exc
        self._client = Groq(api_key=key, timeout=config.timeout_seconds)

    def complete(
        self,
        messages: Sequence[Message],
        schema: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> BaseModel | str:
        """Call Groq, retrying on parse/validation failure up to ``max_parse_retries``."""
        model_name = model or self.config.model
        payload: list[Message] = list(messages)
        if schema is not None:
            payload = [{"role": "system", "content": _schema_instruction(schema)}, *payload]

        attempts = self.config.max_parse_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            estimated = estimate_tokens(payload, self.config.max_tokens)
            self.rate_limiter.acquire(estimated)
            started = time.monotonic()
            kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": payload,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            if schema is not None and self.config.json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self._client.chat.completions.create(**kwargs)
            latency = time.monotonic() - started
            text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            if total_tokens:
                self.rate_limiter.settle(estimated, total_tokens)

            self.call_logger.log(
                backend="groq",
                model=model_name,
                attempt=attempt,
                schema=schema.__name__ if schema else None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=round(latency, 4),
            )

            if schema is None:
                return text
            try:
                return schema.model_validate_json(_extract_json(text))
            except (ValidationError, LLMError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("schema parse failed (attempt %d/%d): %s", attempt, attempts, exc)
                payload = [
                    *payload,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "That response did not validate against the schema. "
                            f"Error: {exc}. Return only a corrected JSON object."
                        ),
                    },
                ]

        raise LLMError(f"could not obtain a schema-valid response after {attempts} attempts: "
                       f"{last_error}")


# --------------------------------------------------------------------------- #
# Echo (offline stub)
# --------------------------------------------------------------------------- #
class EchoClient(LLMClient):
    """Deterministic offline backend for tests and CI. Makes no network calls.

    With a schema it constructs a schema-valid instance by filling required
    fields with type-appropriate defaults; without one it echoes the last user
    message. Canned responses can be supplied to script a specific test.
    """

    def __init__(
        self,
        config: LLMConfig,
        responses: Sequence[BaseModel | str] | None = None,
        call_logger: CallLogger | None = None,
    ) -> None:
        super().__init__(config)
        self._responses = list(responses or [])
        self._index = 0
        self.call_count = 0
        self.call_logger = call_logger

    def complete(
        self,
        messages: Sequence[Message],
        schema: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> BaseModel | str:
        """Return the next canned response, or a synthesised schema-valid object."""
        self.call_count += 1
        if self.call_logger is not None:
            self.call_logger.log(
                backend="echo",
                model=model or self.config.model,
                schema=schema.__name__ if schema else None,
                prompt_tokens=estimate_tokens(messages, 0),
                completion_tokens=0,
                total_tokens=estimate_tokens(messages, 0),
                latency_seconds=0.0,
            )
        if self._index < len(self._responses):
            canned = self._responses[self._index]
            self._index += 1
            return canned
        if schema is None:
            for message in reversed(messages):
                if message.get("role") == "user":
                    return f"echo: {message.get('content', '')}"
            return "echo: (no user message)"
        return schema.model_validate(self._defaults_for(schema))

    @staticmethod
    def _defaults_for(schema: type[BaseModel]) -> dict[str, Any]:
        """Build a minimal payload satisfying the schema's required fields."""
        payload: dict[str, Any] = {}
        for name, field in schema.model_fields.items():
            if not field.is_required():
                continue
            payload[name] = EchoClient._default_value(field.annotation)
        return payload

    @staticmethod
    def _default_value(annotation: Any) -> Any:
        """A neutral, valid-ish value for a field annotation."""
        if annotation is None:
            return None
        if isinstance(annotation, type):
            if issubclass(annotation, BaseModel):
                return EchoClient._defaults_for(annotation)
            if issubclass(annotation, bool):
                return False
            if issubclass(annotation, int):
                return 1
            if issubclass(annotation, float):
                return 1.0
            if issubclass(annotation, str):
                return "echo"
        origin = getattr(annotation, "__origin__", None)
        if origin in (list, tuple, set):
            return []
        if origin is dict:
            return {}
        return "echo"


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_llm_client(config: LLMConfig | None = None, **kwargs: Any) -> LLMClient:
    """Build the backend named in ``configs/agent.yaml`` (or in ``config``)."""
    if config is None:
        agent_config = cast(AgentConfig, load_config("agent"))
        config = agent_config.llm
    if config.provider == "groq":
        return GroqClient(config, **kwargs)
    if config.provider == "echo":
        return EchoClient(config, **kwargs)
    raise LLMError(f"unknown LLM provider {config.provider!r}")
