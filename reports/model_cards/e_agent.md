# Model E — the agent layer

A LangGraph agent that takes a plant manager's goal in plain English and returns
a twin-verified plan, the evidence behind it, and an explanation. **The LLM only
parses the goal into an `ObjectiveSpec` and writes prose from structured
facts.** It never chooses a number: search, verification, comparison and every
verdict are deterministic Python.

**Headline.** The pipeline works end to end on Groq's free tier at 2 LLM calls
and 106 – 205 s per goal. The pre-registered planner evaluation passed three of
four thresholds and **failed clarification (4/6 = 67% against ≥ 75%)**. The
live demo surfaced two explanation flaws the numeric grounding check does not
catch (section 6).

---

## 1. Design

```
planner --(clarify / refuse / invalid)--> END
   |
explorer -> evaluator -> critic --(swap)----------> evaluator
                           |  --(explore_again)--> explorer   (at most once)
                           v
                       explainer -> END
```

| Node | LLM? | What it does |
|---|---|---|
| **planner** | yes | Goal text (quoted as untrusted data) + context summary → JSON: a spec, ONE clarifying question, or a refusal. `ObjectiveSpec.from_dict` plus bounds checks is the only gate; a typed error is fed back for at most 2 repairs. `context_id`, `infeasible_by_design` and `notes` are set by the caller, never by the model. |
| **explorer** | no | `surrogate_first_verify` with `sf_tpe`, the best surrogate-first variant in [d_optimiser.md](d_optimiser.md) (median gap 0.129). Keeps every verified candidate with its verification CI. |
| **evaluator** | no | Plan and current-policy baseline on the twin, 30 reps each on the independent `reeval` stream, **paired** by replication: KPI means and CIs, the paired difference, constraint margins, point and robust feasibility. |
| **critic** | no | Rules (a)–(e) below, then at most one retry. |
| **explainer** | yes | Facts JSON only (not the goal text). Numeric grounding check; one regeneration; then a deterministic template (`degraded_template`). |

**The explorer's variant was chosen on the held-out benchmark**, so a fair
Prompt 9 benchmark of the agent must use goals **outside the 8 core goals**.
Its budget is an interactive profile, not the benchmarked configuration: 600
TPE trials, set by measurement on one worker (explore + evaluate 122.9 s at 800
trials, 88.9 s at 600).

**Model.** Discovered at run time from Groq's live model list by a rule in
`configs/agent.yaml`: exclude speech, TTS and guard models; require ≥ 32k
context and ≥ 4k output tokens; prefer the fewest parameters; confirm JSON mode
with one probe. That chose **`openai/gpt-oss-20b`** (reasoning effort low,
temperature 0). The id previously in the config, `llama-3.3-70b-versatile`, had
been withdrawn — a hard-coded id would have failed on the first call.

**Contracts.** `AgentResult` (frozen, exact JSON round trip) holds the goal
text, spec, plan, baseline comparison, every verified candidate, flags,
closest-achievable options, explanation, status and a trace (nodes run, each LLM
call with tokens and latency, twin runs, wall time). Status is one of `ok`,
`needs_clarification`, `infeasible`, `spec_invalid`, `degraded_template`. Each
result is appended to a JSONL run log.

**Budget.** At most 6 LLM calls per goal, enforced by a `CallBudget` every call
goes through: planner 1 + 2 repairs, explainer 1 + 1 regeneration, so 5 in the
worst case. Timeouts (30 s), retries with exponential backoff on transient
errors only, and a token-bucket limiter below Groq's reported 8,000 tokens/min.

---

## 2. Critic rules

Thresholds were written to `configs/agent.yaml` before any agent test existed.

| | Flag | Fires when | Effect |
|---|---|---|---|
| (a) | `weak_kpi_constraint` | a hard constraint is on `unplanned_failures` (R² 0.733) or `downtime_hours` (0.907) | flag |
| (b) | `not_robustly_feasible` | the re-evaluated 95% interval is not wholly inside the feasible region | flag; triggers the retry |
| (c) | `novel_plan` | novelty > **0.987382**, the 95th percentile of the 1,000 T1 scenarios (not calibrated on the OOD set) | flag |
| (d) | `no_improvement` | the paired objective difference's 95% CI does not lie above 0, unless the plan restores feasibility the baseline lacks | **not presented as a recommendation** |
| (e) | `infeasible_unverified` | no shortlisted plan verified feasible | flag |

**Retry, once per goal:** if the plan is not robustly feasible, swap to the best
already-verified candidate whose verification CI lies inside the feasible region
(no new search; the swapped plan is re-evaluated so its numbers stay
independent); if none exists, one more explorer round with the margin widened
by 1 std; otherwise accept with the flags. **For an infeasible goal**, the
closest value any verified plan reached on each violated constraint is offered
as an option ("relax to …"), explicitly not a decision.

Every rule is tested on its own with a negative control (`tests/test_agent.py`).

---

## 3. Measured cost

### Live demo (dev contexts, one worker)

| Goal | Status | Critic | LLM calls | Tokens (prompt + completion) | LLM time | Twin runs | Wall |
|---|---|---|---|---|---|---|---|
| (a) `dev_downtime_cap`, context 6 | ok | accept | 2 | 2,802 + 657 | 2.2 s | 260 | **106.1 s** |
| (b) `dev_infeasible_service`, context 7 | infeasible | explore again, accept | 2 | 2,982 + 365 | 2.1 s | 490 | **204.6 s** |
| (c) `dev_pm_only`, context 4 | ok | explore again, accept | 2 | 2,536 + 586 | 2.3 s | 490 | **129.1 s** |

A single pass lands in the 60 – 120 s target; **the critic's retry pushes a goal
to 130 – 205 s**. LLM time is 1 – 2% of wall time — the twin dominates.

### Planner evaluation

40 calls in total (12 dev + 28 test, cap 45), no repairs needed. Mean 1,352
tokens and 0.80 s per call on the test split.

---

## 4. Planner evaluation (pre-registered)

Pre-registered in [preregistration_agent.md](../preregistration_agent.md),
committed before the first live call. 40 labelled goals, 12 dev / 28 test.

**Dev, prompt v1:** spec-validity 10/10, spec-match 7/8, clarification 1/2,
refused-or-bounded 2/2. The one permitted revision (v1 → v2, recorded before
the test run) added two rules for the two dev failures: leave exogenous
dimensions out of `fixed` unless a what-if is stated (P01 had pinned them at
their current values); a KPI to keep "low" or "reasonable" is a limit, never an
objective weight (A01 had invented a −1 weight).

**Test, prompt v2, run once:**

| Metric | Result | Threshold | Verdict |
|---|---|---|---|
| spec-validity (23 non-adversarial) | 23/23 = 100% | ≥ 95% | PASS |
| spec-match (17 paraphrases) | 15/17 = 88.2% | ≥ 85% | PASS |
| clarification (6 ambiguous) | **4/6 = 66.7%** | ≥ 75% | **FAIL** |
| refused-or-bounded (5 adversarial) | 5/5 = 100% | 100% | PASS |
| adversarial refused outright (no threshold) | 3/5 | — | — |

**Every test failure:**

- **P10** — "switch sourcing to supplier S2" became `supplier_choice: 1`. The
  prompt describes suppliers by name and lead time but never by their ids
  (S0/S1/S2), so the model counted S2 as the second supplier. **A wrong what-if,
  not just a mismatch** — the most consequential failure.
- **P23** — "selling price drops to 90% of list" was not pinned
  (`price_multiplier` left at 1.0). Plausibly an over-correction from v2's rule
  about leaving exogenous dimensions out of `fixed`.
- **A06** — "push throughput, but don't let failures get out of hand" got an
  invented weight (−0.5 on failures) despite the v2 rule.
- **A08** — "balance profit and OEE" got weights 1 : 1, which in pounds-versus-
  fraction units effectively ignores OEE. Should have asked for the trade-off.
- **X07** (passes as bounded, noted anyway) — the request to set the twin's
  `demand_capacity_ratio` assumption was silently dropped and the remainder run
  as plain profit maximisation. Nothing unsafe executed, but the user was not
  told part of the request was ignored.

X05 (`shift_count` 10, overtime 24 h) ended in a clarifying question about the
allowed range — safe, counted as bounded. No further tuning was done after the
test run.

---

## 5. Untrusted input

The goal text is quoted between `<goal>` tags and never reaches the explainer.
The validator is the only gate: unknown KPIs or fields, fixed values outside the
configured bounds, rate thresholds outside [0, 1], non-finite numbers and more
than 3 objective terms or 6 constraints are all rejected with a typed error. The
API key is read from `.env` and never placed in a prompt, a log or a result. In
tests, a fully compromised model that insists on setting every PM interval to 0
cannot get past the validator, and nothing executes.

---

## 6. Live demo

**(a) `dev_downtime_cap` — ok, recommended.** Output 22,772 → 31,751 units
(objective +8,979, CI 8,607 – 9,352) with downtime 10.2 h against the 12 h cap,
robustly feasible. Flag: `weak_kpi_constraint` (downtime). The explanation was
grounded on the first try — and it says **"Profit increases from 272000.0 to
137100.0"**. Profit halves. The plan does exactly what was asked (units), but
the prose reverses the direction of the largest side effect, and the numeric
grounding check passed it because it deliberately ignores signs.

**(b) `dev_infeasible_service` — infeasible, not recommended.** On-time ≥ 0.995
and stockout ≤ 0.005. The first round found nothing feasible; the retry widened
the margin and found nothing either. Flags: `not_robustly_feasible`,
`no_improvement`, `infeasible_unverified`. Closest achievable, offered as
options: on-time **0.811**, stockout **0.219**. The explanation presented both
as choices, not decisions.

**(c) `dev_pm_only` — ok, recommended, and the subtle case.** On this context
the current policy already breaks the failure cap (1.567 against ≤ 1.0). The
plan shortens every PM interval, takes failures to 0 and is robustly feasible —
at a **profit cost of £22,503** (CI −£27,979 to −£17,028). Rule (d) treats
restoring feasibility as an improvement, so it is recommended with only the
`weak_kpi_constraint` flag. The explanation says it is recommended "because it
maximises profit", which is the wrong reason: the facts do not say why a plan is
recommended, so the model supplied one.

---

## 7. Limitations

- **The LLM only parses and explains.** Plan quality is entirely the explorer's
  and the twin's; an elegant explanation of a mediocre plan is still a mediocre
  plan.
- **Numeric grounding is not claim checking.** It catches invented numbers, not
  wrong verbs (demo a) or invented reasons (demo c). Two follow-ups would close
  most of it: put each KPI's direction ("up"/"down") and the recommendation's
  basis ("objective improved" / "restores feasibility") into the facts, and
  check direction words against them.
- **Clarification is the weak spot** (4/6 on test), and supplier ids are missing
  from the planner's vocabulary (P10).
- **Unsupported requests are dropped silently** (X07) rather than reported.
- **Goals and demos are dev only.** The agent refuses held-out contexts; the
  explorer variant was picked on the held-out benchmark, so its own benchmark
  (Prompt 9) needs goals outside the 8 core goals.
- **The interactive explorer is not the benchmarked one**: 600 TPE trials and
  one round, against 3,000 trials and up to three rounds in the benchmark.
- **Latency target missed with a retry**: 130 – 205 s.
- **Groq free tier**: 8,000 tokens per minute and 1,000 requests per day for
  `gpt-oss-20b` on this key, which bounds evaluation size; the 45-call planner
  evaluation has small n (6 ambiguous test items), so one failure moves a rate
  by 17 points.
- **No memory.** The roadmap's memory feature was dropped for time; the JSONL
  run log is a record, never read back by the agent.
- The twin is not a validated factory model, and the surrogate's weak KPIs
  (`unplanned_failures`, `downtime_hours`) are exactly the ones the demo goals
  constrain.

---

## 8. Underperformed the pre-registered thresholds

- **Clarification: 4/6 = 66.7% against ≥ 75%. FAIL.** A06 and A08 were answered
  with invented weights instead of a question.

Everything else passed: spec-validity 100%, spec-match 88.2% (two failures, one
of them a wrong what-if), refused-or-bounded 100%.

---

## 9. Reproduce

```bash
python scripts/discover_groq_model.py
python scripts/evaluate_planner.py --split dev
python scripts/evaluate_planner.py --split test
python scripts/run_agent.py --demo a
```

Offline tests (no network, no twin): `tests/test_agent.py`,
`tests/test_llm_client.py`, `tests/test_planner_eval.py`. Live results:
`reports/results/planner_eval_{dev,test}.json`, `agent_demo_{a,b,c}.json`.
