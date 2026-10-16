"""Post-benchmark OFFLINE re-check of the 40 stored audit explanations, v1 vs v2.

    python scripts/recheck_explanations_v2.py

NOT a test result and NOT a change to B5. No LLM call is made and the benchmark
audit is not re-run: the 40 explanation texts stored in
reports/results/agent_bench_audit.json are checked by both checker versions
against the facts the explainer saw (rebuilt with the v1 facts builder, as in
the benchmark).

What is stored is each cell's FINAL text -- the passing LLM text, or the
template after a fallback. The failed first and second attempts were not
stored, so this cannot re-score them; that gap is why every attempt is now
recorded.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from twinmind.agents.critic import recommendation_reason
from twinmind.agents.explanation_v2 import audit_v2, passes
from twinmind.agents.tools import audit_explanation, audit_passes, build_facts
from twinmind.eval.registry import results_dir
from twinmind.optim.contracts import ObjectiveSpec
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("recheck_explanations_v2")
CELLS = PROJECT_ROOT / "data" / "processed" / "agent_bench" / "full"
OUT = results_dir() / "explanation_check_v2_offline.json"


def main() -> int:
    audit = json.loads((results_dir() / "agent_bench_audit.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for key, entry in sorted(audit["explainer"].items()):
        result = json.loads((CELLS / f"{key}.json").read_text(encoding="utf-8"))["result"]
        spec = ObjectiveSpec.from_dict(result["spec"])
        evaluation = result["baseline_comparison"]
        facts = build_facts(
            spec,
            result["plan"],
            evaluation,
            result["flags"],
            result["closest_achievable"],
            result["plan"]["recommended"],
            recommendation_reason(spec, evaluation, {f["code"] for f in result["flags"]}),
        )
        v1 = audit_explanation(entry["text"], facts)
        v2 = audit_v2(entry["text"], facts)
        rows.append(
            {
                "cell": key,
                "stored_mode": entry["mode"],
                "v1_pass": audit_passes(v1),
                "v2_pass": passes(v2),
                "v1": v1,
                "v2": v2,
            }
        )
    summary = {
        "label": "post-benchmark offline re-check, not a test result; B5 is unchanged",
        "n": len(rows),
        "v1_pass": sum(r["v1_pass"] for r in rows),
        "v2_pass": sum(r["v2_pass"] for r in rows),
        "v2_fail_reasons": {
            r["cell"]: {k: v for k, v in r["v2"].items() if v} for r in rows if not r["v2_pass"]
        },
        "rows": rows,
    }
    OUT.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    logger.info(
        "stored final texts: v1 passes %d/%d, v2 passes %d/%d",
        summary["v1_pass"],
        summary["n"],
        summary["v2_pass"],
        summary["n"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
