"""Build the dashboard's offline demo cache from EXISTING benchmark outputs.

    python scripts/build_demo_cache.py

No agent run and no LLM call. Each entry is a benchmark agent cell
(data/processed/agent_bench/full) -- a complete AgentResult, run on the gold
spec with no explainer -- joined with the explanation the benchmark audit wrote
for it (reports/results/agent_bench_audit.json) and the goal's
natural-language text (tests/data/agent_bench_goals.yaml). Written to
app/demo_cache/ as small JSON: candidates trimmed to the best five and the
per-evaluation history dropped, with no data/ paths and nothing from .env.

The cached explanations were produced by explainer v1 during the benchmark;
all of them also pass checker v2 (reports/results/explanation_check_v2_offline.json).
"""

from __future__ import annotations

import json
import sys
from typing import Any

from twinmind.agents.bench_goals import load_bench_goals
from twinmind.agents.contracts import AgentResult
from twinmind.agents.explanation_v2 import forbidden_word_errors
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("build_demo_cache")
CELLS = PROJECT_ROOT / "data" / "processed" / "agent_bench" / "full"
AUDIT = PROJECT_ROOT / "reports" / "results" / "agent_bench_audit.json"
OUT = PROJECT_ROOT / "app" / "demo_cache"

#: (goal id, seed, role shown in the UI). Seed 1 throughout; every pick has an
#: LLM-written explanation (none of the benchmark's 5 template fallbacks).
PICKS = [
    ("G07", 1, "A clean feasible goal: no critic flag, accepted first time."),
    (
        "G01",
        1,
        "The critic acted: it swapped plans, and the plan is still flagged as "
        "not robustly feasible.",
    ),
    ("I01", 1, "An infeasible goal: no plan meets the constraints; closest-achievable options."),
    ("N01", 1, "No room: the only lever cannot move the objective, so nothing is recommended."),
    ("AA1", 1, "Ambiguous wording with one reasonable reading."),
    (
        "G09",
        1,
        "A constraint on unplanned failures, a weakly predicted KPI; also flagged as novel.",
    ),
]

FORBIDDEN_STRINGS = ("data/", "data\\", ".env", "gsk_", "GROQ_API_KEY")


def build_entry(
    goal_id: str, seed: int, role: str, goals: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    key = f"{goal_id}__agent__seed{seed}"
    cell = json.loads((CELLS / f"{key}.json").read_text(encoding="utf-8"))
    result = dict(cell["result"])
    explanation = audit["explainer"][key]
    if explanation["mode"] != "llm":
        raise ValueError(f"{key}: the demo cache uses LLM-written explanations only")
    if forbidden_word_errors(explanation["text"]):
        raise ValueError(f"{key}: explanation uses a forbidden word")
    goal = goals[goal_id]
    result["goal_text"] = goal.text
    result["explanation"] = explanation["text"]
    result["explanation_mode"] = "llm"
    result["candidates"] = sorted(result["candidates"], key=lambda c: -c["objective"])[:5]
    trace = dict(result["trace"])
    trace.pop("history", None)
    trace["explanation"] = {
        "source": "agent benchmark audit (explainer v1); passes checker v2 offline",
        "attempts": explanation["attempts"],
    }
    result["trace"] = trace
    AgentResult.from_dict(result)  # must stay a valid AgentResult
    return {
        "id": goal_id.lower(),
        "goal_id": goal_id,
        "seed": seed,
        "category": goal.category,
        "role": role,
        "context_id": goal.context_id,
        "replayed": True,
        "provenance": f"Replayed from the Prompt 9 agent benchmark, goal {goal_id}, seed {seed}; "
        "not a live run.",
        "result": result,
    }


def main() -> int:
    goals = {g.id: g for g in load_bench_goals()}
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for goal_id, seed, role in PICKS:
        entry = build_entry(goal_id, seed, role, goals, audit)
        text = json.dumps(entry, indent=1, ensure_ascii=False)
        bad = [s for s in FORBIDDEN_STRINGS if s in text]
        if bad:
            raise ValueError(f"{goal_id}: cache entry contains {bad}")
        path = OUT / f"{entry['id']}.json"
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
        index.append(
            {k: entry[k] for k in ("id", "goal_id", "seed", "category", "role", "context_id")}
            | {"goal_text": entry["result"]["goal_text"], "file": path.name}
        )
        logger.info("%s: %d bytes", path.name, path.stat().st_size)
    (OUT / "index.json").write_text(
        json.dumps({"entries": index}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
