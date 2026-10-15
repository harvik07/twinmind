"""Pick the agent's Groq model from the live model list, by the rule in configs/agent.yaml.

    python scripts/discover_groq_model.py

Lists Groq's models (a metadata call, no tokens), ranks the eligible ones by the
selection rule, and probes the first with ONE tiny JSON-mode completion. Prints
the chosen id and the rate limits Groq reports for it, read from the response
headers rather than assumed. The id is then recorded in configs/agent.yaml by
hand; at run time :func:`twinmind.agents.graph.resolve_model` re-checks it
against the live list and re-discovers if it has been withdrawn.

Never prints the API key.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from twinmind.agents.llm_client import (
    ModelRule,
    discover_model,
    extra_params_for,
    load_api_key,
    rank_models,
)
from twinmind.utils.config import load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("discover_groq_model")


def main() -> int:
    config: Any = load_config("agent")
    llm = config.llm
    from groq import Groq

    client = Groq(
        api_key=load_api_key(llm.api_key_env, PROJECT_ROOT / ".env"),
        timeout=llm.timeout_seconds,
        max_retries=0,
    )
    rule = ModelRule(
        exclude_patterns=tuple(llm.selection.exclude_patterns),
        min_context_window=llm.selection.min_context_window,
        min_completion_tokens=llm.selection.min_completion_tokens,
    )
    ranked = rank_models(client.models.list().data, rule)
    logger.info("eligible, fastest first: %s", ranked)
    limits: dict[str, str] = {}

    def probe(model_id: str) -> bool:
        params = extra_params_for(model_id, llm.extra_params_by_prefix)
        raw = client.chat.completions.with_raw_response.create(
            model=model_id,
            messages=[
                {"role": "system", "content": 'Reply with the JSON object {"ok": true} only.'},
                {"role": "user", "content": "ping"},
            ],
            response_format={"type": "json_object"},
            max_tokens=256,
            temperature=0.0,
            **params,
        )
        for name in (
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
        ):
            limits[name] = raw.headers.get(name, "?")
        completion = raw.parse()
        usage = completion.usage
        logger.info(
            "probe %s: %d prompt + %d completion tokens",
            model_id,
            usage.prompt_tokens,
            usage.completion_tokens,
        )
        try:
            return json.loads(completion.choices[0].message.content or "").get("ok") is True
        except json.JSONDecodeError:
            return False

    chosen = discover_model(client, rule, probe)
    logger.info("chosen model: %s", chosen)
    logger.info("rate limits reported by Groq: %s", limits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
