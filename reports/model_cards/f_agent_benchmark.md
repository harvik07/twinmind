# Model F — the agent benchmark

The agent (Prompt 8) measured on **20 new goals** it was never tuned on, against
CMA-ES driving the twin directly at equal wall-clock, with the critic ablated,
under a pre-registration committed before any run
([preregistration_agent_bench.md](../preregistration_agent_bench.md), Amendment
A.1 included).

**Headline.** B1, B3 and B4 **PASS**; B2 and B5 **FAIL**.

- The agent's plans are much closer to the reference than twin-only's (mean gap
  0.196 against 0.531).
- The critic removes 11 of 12 bad recommendations.
- Every impossible and pointless goal is handled correctly.
- **B2 fails by one cell** (95.7% against 96.9%).
- **B5 fails on the explanation clause**: 35/40, where 40/40 was required.

---

## 1. Design

**Why new goals.** The agent's optimiser variant (`sf_tpe`) was chosen on the 8
core held-out goals, so they could not be reused. The 20 goals sit on T1
contexts 8–19, which no dev, core or reserve goal uses
(`tests/data/agent_bench_goals.yaml`).

**Goal mix: 11 feasible / 4 infeasible / 3 no-room / 2 ambiguous-but-answerable,
not the 12 / 3 / 3 / 2 first specified.** Every label was checked on the twin at
a small budget before pre-registration:

- **G11 relabelled infeasible.** It was written as feasible, but the offshore
  supplier's 21-day lead time keeps on-time delivery below 0.70 even for the
  current policy.
- **N01 and N02 redesigned.** Their first versions turned out to have room:
  safety stock moved OEE by +0.040, and the reorder point cut downtime by 6.8 h.
- **All three no-room goals are therefore PM-interval levers**, each with
  **exactly one free dimension** (N01: profit via M5's interval; N02: on-time
  via M4's; N03: on-time via M5's). That is a narrow kind of "no room".

**Arms**, all on the **gold** spec, so optimisation is measured separately from
parsing. Seeds 1 and 2; every final plan is re-evaluated at 30 reps on the
independent reeval stream, paired with the baseline.

| Arm | What it is |
|---|---|
| `agent` | Explorer, evaluator and critic with its one retry. `sf_tpe` at **600 TPE trials — the interactive profile, not the 3,000-trial configuration benchmarked in d_optimiser.md**; K = 20; 10 verification reps; one round. No LLM. |
| `agent_nocritic` | The explorer's first plan, before the critic, from the same trace. |
| `twin_only` | CMA-ES on the twin, 5 reps per evaluation, for **exactly the agent's measured wall-clock** on that goal and seed, restarted on early convergence. |
| `baseline` | The current policy. |

**Smoke test, before the main run** (2 dev goals, seed 1, 2 workers): 4 cells,
0 failures, 3.7 min. Agent cells took 76.9 s and 105.5 s; twin-only searched
70.4 s and 99.5 s against agent budgets of 70.2 s and 98.8 s; peak memory per
worker was 0.42–0.45 GB resident and 1.22–1.25 GB committed.

**Main run.** 80 cells on 3 workers in 39.7 min, plus a 125 s re-run (below).
Agent wall-clock per cell: median 68 s, range 46–166 s.

---

## 2. The A.1 incident

The main run wrote **74 of 80** cells. All 6 failures were `twin_only` on N01,
N02 and N03, both seeds, each with
`ValueError: not yet initialized (dimension needed)`.

- **Root cause: a pycma 4.5.0 bug that only bites in one dimension.** With
  bounds, pycma caps each coordinate's step size at a third of the range. When a
  step crosses that cap, `DiagonalDecoding.set_i` treats a size-1 scaling vector
  as "not yet initialized" and raises. In one dimension an initialised vector
  is size 1 too; in two or more dimensions the cap is simply applied.
- **Why only N01–N03.** They are the only one-dimensional goals. Their flat
  objectives made CMA-ES stop and restart many times within its budget, so some
  seed always crossed the cap.
- **Fix** (Amendment A.1, a decision the pre-registration was silent on): treat
  that error as the end of the CMA-ES run, keep every point already evaluated,
  and restart under the pre-registered rule. Behaviour in two or more
  dimensions is unchanged.
- **No outcome was inspected before the fix.** The diagnosis used the code, the
  gold specs and offline reproductions.
- **Before the change**, the sha256 of the manifest and all 74 cells was
  recorded (`reports/results/agent_bench_cell_sha256_before_A1.txt`). **After
  it**, and again at analysis time, all 75 were byte-identical.
- **Only the 6 failed cells were re-run** (125 s, 0 failures). The original
  failure records remain in `data/processed/agent_bench/full/failures/`.

**Even after the fix, twin-only on N01–N03 is many short restarts**: 64
evaluations per cell across 9–16 restarts, about 4–7 evaluations each. CMA-ES
is not designed for one dimension, so read twin-only's numbers on those goals
with that in mind.

---

## 3. Results

| | Verdict | Deciding number |
|---|---|---|
| **B1** agent non-inferior to twin-only on normalised gap | **PASS**, and superior | mean(agent − twin-only) = **−0.335**, 95% CI −0.520 to −0.141, 13 goals (bound +0.10) |
| **B2** robust-feasibility rate of recommended plans, agent ≥ twin-only | **FAIL** | agent **22/23 = 95.7%** against twin-only **31/32 = 96.9%** |
| **B3** critic on vs off: fewer bad recommendations | **PASS** | agent **1** against no-critic **12** |
| **B4** all 4 infeasible goals declared infeasible with options; ≥ 2 of 3 no-room goals flagged | **PASS** | 4/4 infeasible on both seeds; 3/3 no-room flagged and not recommended on both seeds |
| **B5** planner spec-match ≥ 85% AND 100% of explanations pass | **FAIL** | planner **20/20**; explanations **35/40** |

### B1 — gap to the reference

The reference is the best re-evaluated feasible plan across all arms and both
seeds, so **a gap is a lower bound on the true gap, not a distance to an
optimum**. Per goal (mean of the 2 seeds; 0 = the reference, 1 = no better than
the current policy):

| Goal | agent | agent_nocritic | twin_only |
|---|---|---|---|
| G01 | 0.064 | 0.092 | 0.654 |
| G02 | **0.427** | 0.253 | **0.109** |
| G03 | 0.017 | 0.017 | 0.543 |
| G04 | **0.962** | 0.962 | **0.722** |
| G05 | 0.156 | 0.156 | 0.314 |
| G06 | 0.157 | 0.157 | 0.653 |
| G07 | 0.037 | 0.037 | 0.821 |
| G08 | 0.174 | 0.174 | 0.532 |
| G09 | 0.005 | 0.005 | 0.052 |
| G10 | 0.003 | 0.003 | 0.503 |
| G12 | 0.401 | 0.401 | 1.242 |
| AA1 | 0.138 | 0.138 | 0.704 |
| AA2 | 0.003 | 0.003 | 0.055 |
| **mean** | **0.196** | **0.185** | **0.531** |

Twin-only wins on **G02** and **G04**; the agent wins everywhere else.
Twin-only's G12 plan is feasible but worse than the current policy (gap 1.242).

**Unstable denominators (flag is post-hoc; the pre-registered rule is
unchanged).** Some goals' reference beats the baseline by less than twice the
baseline's own 95% CI half-width:

- **G04 seed 2: exactly 0.** The reference *is* that seed's baseline, so the
  pre-registered rule excludes the seed and G04's gap rests on seed 1, whose
  ratio is only 1.18.
- **G12 seed 2: 1.31. G07 seed 2: 1.46. G08 seed 2: 1.64.**

*Post-hoc sensitivity, no verdict:* without those four goals the B1 difference
is −0.291 (CI −0.476 to −0.090, n = 9), so the conclusion does not depend on
them.

![Cost-quality](../figures/f_bench_cost_quality.png)
![Per-goal gap](../figures/f_bench_per_goal_gap.png)

### B2 — robust feasibility of recommended plans

The agent recommended 23 plans and 22 were robustly feasible; the exception is
G01 seed 1. Twin-only "recommended" 32 and 31 were robust.

**The comparison is not like for like.** Twin-only has no critic: under its
pre-registered rule it recommends every plan that was feasible on its own
search stream. **17 of its 32 recommended plans do not beat the current
policy.** Those are trivially robust, and B2 rewards them. The agent declined
17 plans: all 8 infeasible-goal cells, all 6 no-room cells, both G04 cells and
G12 seed 2. B2 still fails as registered.

### B3 — the critic

Bad means recommended AND (not robustly feasible OR not beating the current
policy). The no-critic arm made **12** bad recommendations:

- all 6 no-room cells (no improvement);
- both G04 cells and G12 seed 2 (no improvement);
- G01 seed 1, G02 seed 2 and G10 seed 2 (not robust).

The critic left **1**: G01 seed 1, which was still not robust after its one
retry. Critic actions across the 40 cells: 29 accept, 10 explore again then
accept, 1 swap then accept.

The critic cost gap on **G02** (0.427 with it against 0.253 without): its retry
on seed 2 traded objective for robustness.

### B4 — impossible and pointless goals

All 4 infeasible goals (I01–I03 and G11) ended `infeasible` on both seeds, each
with closest-achievable options. All 3 no-room goals were flagged
`no_improvement` and not recommended on both seeds.

The no-room clause passing says less than it seems: all three goals are
PM-interval levers with one free dimension, on contexts where those intervals
barely act.

![Critic flags](../figures/f_bench_critic_flags.png)

### B5 — parsing and explanation

**Planner v3: 20/20 spec-matches.** Every NL goal produced a spec; there were 0
clarifications, 0 refusals and 0 invalid specs. **The benchmark has no
clarification-needed goal**, so the earlier 4/6 clarification miss (Prompt 8
test split) is **not retested here**.

**Explainer on the 40 agent results**, three ways:

| Count | Passed |
|---|---|
| First attempt | **30/40** |
| LLM text, within the one permitted regeneration — **the pre-registered criterion** | **35/40** |
| Including the template fallback | 40/40 |

The pre-registration counts a template fallback as a failure, so **B5 fails on
the explanation clause: 35/40 against 100%.** The 5 fallbacks were AA2 seed 1,
G01 seed 2, G03 seed 1, G10 seed 1 and N02 seed 1. The audit used 70 of its 120
LLM calls.

**Post-hoc, no verdict: why the 10 retried or fallen-back explanations failed
their first attempt.** By reading the checker's 25 recorded first-attempt
reasons, with the checker unchanged:

| Kind | Items | What it was |
|---|---|---|
| **Real error** | 3 | "downtime hours **fall** from 14.69 to 20.05" (AA2 s2); "downtime hours **fall** from 10.66 to 12.02" (G02 s1); and a garbled self-correction in the same G02 s1 text |
| Checker false flag: scientific notation | 8 | N02 seed 1 copied facts written like `-1.208e-05`; the number check splits them into "1.208" and "05" |
| Checker false flag: explanatory phrasing | 12 | e.g. "a decrease …, which is worse when higher is better". The direction stated is correct; the "higher" belongs to the phrase. A few attach a list's direction word to the wrong KPI. |
| Undeterminable | 2 | the stored 80-character excerpt is too short to tell |

- **Negation false flags: 0.** I predicted these in Amendment A.1; no failure
  involved a delta of exactly zero, so the prediction did not hold.
- **All 5 template fallbacks began from false flags only.** The retries'
  reasons were not stored, so why each retry also failed is unknown.
- **Only 2 of the 10 cells show a clear real error at the first attempt.**

The B5 explanation verdict stays **FAIL**. The evidence suggests it is mostly a
**checker precision problem** — the explainer's own prompt invites "worse when
higher is better" phrasing, and the facts allow scientific notation — rather
than fabricated claims.

---

## 4. Twin-component ablation (exploratory, pre-declared)

Each agent final plan and its baseline were re-evaluated at 30 paired reps with
`hazard_source` switched from `learned` to `fixed_mtbf`, a twin-config value.

- **Rankings mostly hold.** Plan minus current policy keeps its sign in **37 of
  40** cells. "Beats the current policy" changes in **1** cell: I02 seed 2, an
  infeasible goal.
- **Levels move a lot.** The baseline's unplanned failures go from 0.66 to
  **2.10** (×3.2), downtime rises **+3.5 h**, and profit falls **£6.1k**.
- **Feasibility flips in 7 cells**: AA2 (both seeds), G01 (both), G02 seed 1
  and G09 (both). **Every one is a goal constraining downtime or failures**,
  feasible under the learned hazard and infeasible under fixed MTBF.

The agent's feasibility verdicts on weak-KPI goals therefore depend on the
health model's assumption, which is the risk the `weak_kpi_constraint` flag
names.

**Quantile vs point demand: not run.** The twin has no point-demand mode. Its
only lever, `demand_spread_multiplier = 0`, collapses each *randomly sampled*
path to a constant at that path's own median, which is not the forecast's
median path. Producing point demand would need a code change.

---

## 5. Underperformed the spec

- **B2 FAIL**: 95.7% against 96.9%. That is one cell, but a failure as
  registered, and the comparison favours an arm with no critic (section 3).
- **B5 FAIL** on the explanation clause: 35/40 against 100%. Most first-attempt
  failures were checker false flags, which is itself a shortfall: the checker
  shipped in Prompt 9 is too imprecise for a 100% bar.
- **The goal mix deviates**: 11/4/3/2 instead of 12/3/3/2, after G11 was
  relabelled.
- **All three no-room goals are one-dimensional PM levers**, which makes B4's
  no-room clause a narrow test.
- **The A.1 incident**: 6 cells crashed on a library bug the smoke test could
  not reach (the smoke goals had many free dimensions). They were fixed and
  re-run under an amendment.
- **The planner audit has no clarification-needed goals**, so the earlier 4/6
  clarification miss is not retested.
- **The agent ran 600 TPE trials, not the 3,000 benchmarked** in d_optimiser.md,
  and the benchmark measured that interactive configuration.
- **The critic hurt one goal** (G02: gap 0.253 → 0.427) and **twin-only beat the
  agent on two** (G02, G04).
- **Point-demand ablation not run** (unsupported without code changes).

---

## 6. Limitations

- **Small sample.** 20 goals, 13 of them gap goals, and 2 seeds. Per-goal
  results come from 2 replicates; G04's gap rests on one seed.
- **The reference is a lower bound**: the best plan any arm found, not the
  optimum.
- **The optimiser was chosen on the core goals** and is run here at a reduced
  budget.
- **Twin-only on N01–N03 is effectively many short restarts.**
- **Free-tier LLM** (`openai/gpt-oss-20b`; 8,000 tokens per minute, 1,000
  requests per day). The audit is one run per item, and its numbers move by 2.5
  points per explanation.
- **Hazard dependence.** Feasibility on the weak-KPI goals depends on the
  hazard model (section 4).
- **The twin is not a validated model of a real factory.** Every number here is
  a statement about the simulator.

---

## 7. Reproduce

```bash
python scripts/run_agent_bench.py --workers 3
python scripts/run_agent_bench.py --audit
python scripts/analyse_agent_bench.py
python scripts/ablate_agent_bench.py --workers 3
```
