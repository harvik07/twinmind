"""Live planner evaluation on Groq, under a hard cap of 45 calls in total.

    python scripts/evaluate_planner.py --split dev
    python scripts/evaluate_planner.py --split test

Pre-registered in reports/preregistration_agent.md, which was committed before
the first live call. Each item runs the production planner (one call plus at
most two repairs). A ledger in reports/results/planner_eval_ledger.json counts
every call across both splits; an item whose worst case would breach the
45-call cap gets fewer repairs, and an item with no calls left is scored as a
failure ("not run") rather than skipped silently.

Only the planner runs here: no twin, no explorer, no explainer. The context
summary is the static one, so no data is loaded. Never prints the API key.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from twinmind.agents.graph import groq_client
from twinmind.agents.llm_client import CallBudget
from twinmind.agents.planner import plan_goal, static_context_summary
from twinmind.agents.planner_eval import aggregate, load_items, score_item
from twinmind.eval.registry import results_dir
from twinmind.utils.config import load_config
from twinmind.utils.logging import get_logger

logger = get_logger("evaluate_planner")

TOTAL_CALL_CAP = 45
#: The Prompt 8 evaluation's ledger. Its test split may never run again.
ORIGINAL_LEDGER = results_dir() / "planner_eval_ledger.json"
CONTEXT_ID = 0


def _ledger(path: Path, cap: int) -> dict[str, Any]:
    if path.exists():
        return dict(json.loads(path.read_text(encoding="utf-8")))
    return {"cap": cap, "calls_used": 0, "runs": []}


def main() -> int:
    parser = argparse.ArgumentParser(description="Live planner evaluation.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--prompt-version", default="v1", help="Recorded with the results; see the card."
    )
    parser.add_argument(
        "--ledger",
        default=ORIGINAL_LEDGER.name,
        help="Call ledger in reports/results; a later check gets its own.",
    )
    parser.add_argument("--cap", type=int, default=TOTAL_CALL_CAP, help="Calls this ledger allows.")
    parser.add_argument("--suffix", default="", help="Appended to the results file name.")
    args = parser.parse_args()

    config: Any = load_config("agent")
    ledger_path = results_dir() / args.ledger
    ledger = _ledger(ledger_path, args.cap)
    original = _ledger(ORIGINAL_LEDGER, TOTAL_CALL_CAP)
    if args.split == "test" and any(r["split"] == "test" for r in original["runs"]):
        raise SystemExit("the test split has already been run; results stand as they fell")
    client = groq_client(config)
    items = [i for i in load_items() if i.split == args.split]
    per_item_max = 1 + config.budget.max_spec_repairs
    rows = []
    started = time.perf_counter()
    for item in items:
        remaining = args.cap - ledger["calls_used"]
        if remaining <= 0:
            rows.append({**score_item(item, "not_run", None), "calls": 0, "note": "call cap"})
            logger.warning("%s: not run, the %d-call cap is exhausted", item.id, args.cap)
            continue
        budget = CallBudget(client, min(per_item_max, remaining))
        outcome = plan_goal(
            item.text,
            CONTEXT_ID,
            static_context_summary(CONTEXT_ID),
            budget,
            min(config.budget.max_spec_repairs, budget.max_calls - 1),
            config.planner.max_objective_terms,
            config.planner.max_constraints,
        )
        ledger["calls_used"] += budget.used
        row = score_item(item, outcome.kind, outcome.spec)
        row.update(
            {
                "text": item.text,
                "calls": budget.used,
                "repairs": outcome.repairs,
                "prompt_tokens": sum(r.prompt_tokens for r in budget.records),
                "completion_tokens": sum(r.completion_tokens for r in budget.records),
                "latency_seconds": sum(r.latency_seconds for r in budget.records),
                "spec": outcome.spec.to_dict() if outcome.spec is not None else None,
                "question": outcome.question,
                "reason": outcome.reason,
            }
        )
        rows.append(row)
        logger.info(
            "%s %-11s -> %-8s calls %d  %s",
            item.id,
            item.category,
            outcome.kind,
            budget.used,
            {k: row[k] for k in ("valid", "match", "clarified", "refused_or_bounded") if k in row},
        )
    metrics = aggregate(rows)
    ledger["runs"].append(
        {
            "split": args.split,
            "prompt_version": args.prompt_version,
            "model": client.model,
            "calls": sum(r["calls"] for r in rows),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8", newline="\n")
    out = results_dir() / f"planner_eval_{args.split}{args.suffix}.json"
    out.write_text(
        json.dumps(
            {
                "split": args.split,
                "prompt_version": args.prompt_version,
                "model": client.model,
                "metrics": metrics,
                "items": rows,
                "calls_this_run": sum(r["calls"] for r in rows),
                "calls_used_total": ledger["calls_used"],
                "wall_seconds": time.perf_counter() - started,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for name, m in metrics.items():
        logger.info(
            "%-20s %s/%s = %s  %s",
            name,
            m["hits"],
            m["n"],
            f"{m['rate']:.3f}" if m["rate"] is not None else "n/a",
            m.get("verdict", ""),
        )
    logger.info(
        "calls this run %d, total %d of %d",
        rows and sum(r["calls"] for r in rows),
        ledger["calls_used"],
        args.cap,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
