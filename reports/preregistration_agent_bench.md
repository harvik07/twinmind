# Pre-registration — the agent benchmark (Prompt 9)

**Committed before any benchmark run.** Fixes the goals, arms, metrics,
hypotheses and decision rules. Nothing — goal, threshold, prompt, critic rule,
budget — is edited after a benchmark result is seen; failures are reported as
failures.

---

## Why a new benchmark

The agent's optimiser variant (`sf_tpe`) was **chosen on the 8 core held-out
goals** (reports/model_cards/d_optimiser.md). Measuring the agent on those goals
would reuse the data that picked it. So this benchmark uses **20 new goals on
T1 contexts 8–19**, which no dev, core or reserve goal uses and which sit inside
the agent's dev-only guard.

## Goals

`tests/data/agent_bench_goals.yaml` (sha256 prefix **`6969edd903809963`**): 20
natural-language goals, each with a gold `ObjectiveSpec`.

| Category | n | Goals |
|---|---|---|
| feasible | 11 | G01–G10, G12 — varied objectives (profit, units, unit cost, OEE, downtime, a weighted pair), frozen-dimension goals (speed only, inventory only, labour only), what-ifs (supplier S1, price +10%), and four constraints on the surrogate's weak KPIs (G01, G02, G04, G09; G10 minimises downtime) |
| infeasible by design | 4 | I01, I02, I03, G11 |
| no room (the only free lever cannot move the objective) | 3 | N01, N02, N03 |
| ambiguous but answerable | 2 | AA1, AA2 |

**Changes made before this pre-registration, from the twin check** (`scripts/
check_bench_goals.py`; Sobol over the surrogate, 4,000 points, shortlist 10 at
5 reps, up to 2 rounds; chosen plan and baseline at 30 paired reps on reeval;
results in `reports/results/agent_bench_goal_check.json`):

- **G11** was written as feasible; the check found no feasible plan (the
  offshore supplier's 21-day lead time keeps on-time delivery below 0.70 even
  for the current policy). **Relabelled infeasible.** The mix is therefore
  **11 / 4 / 3 / 2**, not the 12 / 3 / 3 / 2 first specified.
- **N01** ("maximise OEE changing only safety stock") and **N02** ("minimise
  downtime changing only the reorder point") failed the no-room check — both
  levers moved their objective significantly. **Redesigned** as single
  preventive-maintenance levers (N01: profit via M5's interval; N02: on-time via
  M4's), screened on the twin and then confirmed by the check. All three
  no-room goals are now PM-interval levers — a narrow kind of "no room".

After these changes, **all 20 labels were confirmed** by the check.

## Arms

All arms run on the **gold** spec, so optimisation is measured separately from
parsing. Optimiser seeds **1 and 2**.

| Arm | What it is |
|---|---|
| `agent` | The graph from the explorer on: `surrogate_first_verify` with **`sf_tpe`, 600 TPE trials** (the interactive profile, not the 3,000-trial benchmarked configuration), shortlist K = 20, verification 10 reps on the search stream, one round, margin 1.0 std; the paired 30-rep evaluator; the critic with its one retry (swap to a robust verified candidate, else one more round at margin + 1 std). No LLM. Agent config hash **`df871356060f978a`**. |
| `agent_nocritic` | The explorer's **first** chosen plan, before any critic selection or retry — read from the agent cell's trace (same search, no extra cost). |
| `twin_only` | CMA-ES on the twin, 5 reps per evaluation, search stream, for **exactly the agent cell's measured wall-clock** on that goal and seed, one worker; restarted with a new seed if it converges early; best point by feasibility-first order. |
| `baseline` | The current policy. |

**Every final plan is re-evaluated at 30 reps on the independent `reeval`
stream**, paired by replication with the baseline (goal seed = optimiser seed).
Twin config hash `d38e63f94f8f0524`, surrogate hash `9af34fae63a00509`.

**"Recommended"**, per arm: `agent` — the critic's verdict; `agent_nocritic` —
the explorer's first-round plan whenever it verified feasible on the search
stream; `twin_only` — its best point whenever it was feasible on its own search
stream.

## Metrics

- **Normalised gap** = (reference − plan) / (reference − baseline), with
  objectives from the 30-rep re-evaluation. **Reference** = the best
  independently re-evaluated point-feasible plan across all arms and both seeds
  for that goal — **a lower bound on the true gap, not an oracle**. A plan that
  is infeasible on re-evaluation scores gap = 1 (keep the current policy). A
  goal whose reference does not exceed its baseline has an undefined gap and is
  excluded from gap aggregates (reported per goal).
- **Gap goals** = categories `feasible` and `ambiguous_answerable` with a
  defined gap. Per goal, an arm's gap is the mean over its 2 seeds.
- Bootstraps resample **goals**, 10,000 draws, percentile 95% intervals, fixed
  seed.

## Hypotheses and decision rules

| | Hypothesis | Decision rule |
|---|---|---|
| **B1** | The agent is non-inferior to `twin_only` on normalised gap | PASS if the paired bootstrap **upper** 95% bound of mean(agent − twin_only) over gap goals is **< +0.10**. Superiority (upper bound < 0) reported separately. |
| **B2** | Recommended plans are robustly feasible at least as often | PASS if the robust-feasibility rate (on re-evaluation) of recommended plans, over all 20 goals × 2 seeds, is **agent ≥ twin_only**. |
| **B3** | The critic prevents bad recommendations | PASS if the agent has **strictly fewer** "bad" outcomes than `agent_nocritic`. Bad = recommended AND (not robustly feasible on re-evaluation OR not beating the baseline, i.e. the paired objective CI does not lie above 0 and the plan does not restore feasibility the baseline lacks). Counts reported. |
| **B4** | Impossible and pointless goals are handled | PASS if **all 4 infeasible goals** end `infeasible` with non-empty closest-achievable options **on both seeds**, AND **at least 2 of the 3 no-room goals** are flagged `no_improvement` and not recommended **on both seeds**. |
| **B5** | Parsing and explanation hold up | PASS if planner v3 spec-match is **≥ 85%** on the 20 NL goals AND **100%** of the agent results' explanations pass the number, direction and reason checks (within the one permitted regeneration; a template fallback counts as a failure). |

## NL audit (the only LLM use)

- **Planner v3** (sha256 prefix `e576f3ce4a2b27d1`; v2 plus the plant's machine
  and supplier ids, nothing else) on all 20 NL goals, one run each with at most
  2 repairs; spec-match against the gold spec by the rule in
  `src/twinmind/agents/planner_eval.py`.
- **Explainer** on each of the 40 agent results (goal × seed), facts rebuilt
  from the result; first-attempt and final audit outcomes recorded.
- **Hard cap: 120 LLM calls** for the whole audit (worst case 20 × 3 + 40 × 2 =
  140, so the cap can bind; an item with no call left is scored as a failure).

## Procedure and cost

```bash
python scripts/run_agent_bench.py --workers 3
python scripts/run_agent_bench.py --audit
```

Smoke (2 dev goals, seed 1, 2 workers): 4 cells in 3.7 min; agent cells 76.9 s
and 105.5 s; twin_only searched 70.4 s and 99.5 s against agent budgets of 70.2 s
and 98.8 s; peak memory 0.42–0.45 GB resident, 1.22–1.25 GB committed per
worker. Estimate for the 80 cells at 150 s per agent cell: **1.16 h on 3
workers cool, ~2.1 h at the measured 1.8× hot slowdown** (2 workers: 1.73 /
3.12 h). The audit is rate-limited (6,000 tokens per minute): roughly 20–30 min.

## Exploratory, declared in advance

A twin-component ablation of the agent's chosen plans and the baseline (hazard
source learned vs fixed MTBF; quantile vs point demand if supported without
code changes) is **exploratory**: reported, never used for a verdict.

## Known limitations, stated in advance

20 goals and 2 seeds give wide intervals; the reference is a lower bound; the
agent's optimiser was chosen on the core goals; the agent runs a reduced
interactive budget; the LLM is a free-tier model (`openai/gpt-oss-20b`) with
8,000 tokens per minute and 1,000 requests per day; all three no-room goals are
PM-interval levers.
