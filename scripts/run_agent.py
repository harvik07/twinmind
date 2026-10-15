"""Run the agent on one goal, live (Groq + the real twin, one worker).

    python scripts/run_agent.py --goal "Maximise profit ..." --context 0
    python scripts/run_agent.py --demo a        # the three Prompt 8 demo goals

Dev contexts only: the agent refuses a held-out context id. Each result is
appended to the JSONL run log in configs/agent.yaml; a demo also writes its
full AgentResult to reports/results/agent_demo_<name>.json. Never prints the
API key.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from twinmind.agents.graph import AgentDeps, groq_client, run_agent, twin_toolkit
from twinmind.eval.registry import results_dir
from twinmind.utils.config import load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("run_agent")

#: The live demo: dev goals on their own dev contexts. No held-out or reserve goal.
DEMOS: dict[str, dict[str, Any]] = {
    "a": {
        "label": "a normal constrained goal (dev_downtime_cap)",
        "context": 6,
        "goal": "Produce as many units as possible while keeping total machine downtime "
        "within 12 hours over the planning horizon.",
    },
    "b": {
        "label": "the infeasible-by-design dev goal (dev_infeasible_service)",
        "context": 7,
        "goal": "Maximise profit with on-time delivery of at least 99.5% and a stockout "
        "probability of no more than 0.5%.",
    },
    "c": {
        "label": "a goal where the optimiser may not beat the current policy (dev_pm_only)",
        "context": 4,
        "goal": "Only change the preventive-maintenance intervals. Maximise profit, with "
        "at most one unplanned failure.",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent on one goal.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--goal")
    group.add_argument("--demo", choices=sorted(DEMOS))
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument(
        "--suffix",
        default="",
        help="Appended to the demo's output name, so a re-run keeps earlier evidence.",
    )
    args = parser.parse_args()
    # The model writes typographic characters (e.g. narrow no-break spaces) that
    # a Windows cp1252 console cannot encode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    config: Any = load_config("agent")
    goal, context = (
        (DEMOS[args.demo]["goal"], DEMOS[args.demo]["context"])
        if args.demo
        else (args.goal, args.context)
    )
    deps = AgentDeps(
        config=config,
        llm=groq_client(config),
        toolkit=twin_toolkit(config),
        run_log=PROJECT_ROOT / config.run_log,
    )
    result = run_agent(goal, context, deps)
    if args.demo:
        out = results_dir() / f"agent_demo_{args.demo}{args.suffix}.json"
        out.write_text(
            json.dumps({"demo": DEMOS[args.demo], "result": result.to_dict()}, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        logger.info("wrote %s", out)
    trace = result.trace
    logger.info("status %s | flags %s", result.status, result.flag_codes)
    logger.info(
        "nodes %s | critic %s | LLM calls %d (%d prompt + %d completion tokens, %.1fs) | "
        "twin runs %d | wall %.1fs",
        trace["nodes_run"],
        trace["critic_actions"],
        trace["llm_calls_total"],
        trace["prompt_tokens"],
        trace["completion_tokens"],
        trace["llm_latency_seconds"],
        trace["twin_runs"],
        trace["wall_seconds"],
    )
    print(result.explanation or result.question or result.refusal or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
