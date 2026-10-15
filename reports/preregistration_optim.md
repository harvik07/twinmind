# Pre-registration — the optimisation benchmark

**Committed before any held-out goal has been run.** Written so the hypotheses,
the metrics and the verdict rule are fixed in advance and cannot drift to fit
whatever the results turn out to be.

---

## What has and has not been looked at

| | |
|---|---|
| **Tuned on** | the 8 **dev** goals (T1 contexts 0–7) only |
| **Never run** | the 12 **held-out** goals (T1 contexts 20–32) |
| Tuned hyperparameters | surrogate safety margin (`margin_std`), shortlist size `K`, minimum diversity distance |
| Fixed in advance, not tuned | novelty `k = 10`, the AUROC gate of 0.75, every budget, the 3 optimiser seeds |

The dev and held-out goals use **disjoint T1 contexts**, enforced by
`load_goal`, so a goal cannot be nearly the same problem on both sides.

---

## Hypotheses

### H1 — surrogate-first gets close to the best plan anyone found

> The **median normalised gap to `best_known`** over the held-out **feasible**
> goals is **≤ 5%**.

Normalised gap for a goal is

```
gap = (objective(best_known) − objective(method)) / (objective(best_known) − objective(baseline))
```

where `baseline` is the **default Scenario on that goal's context** — the
current operating policy — and every objective is the **independent 30-rep
re-evaluation on the "reeval" stream**. A gap of 0 means "as good as the best
plan found by anything"; 1 means "no better than doing nothing".

`best_known` is deliberately **not** called an oracle: it is the best
twin-verified plan across all methods plus a 500,000-evaluation surrogate sweep
whose top 100 are verified. A method may beat it, giving a **negative** gap.
That will be reported as a negative number, not clipped at zero.

### H2 — it beats twin-only search at a fraction of the twin cost

> Surrogate-first **matches or beats** twin-only TPE and twin-only CMA-ES at
> their 300-evaluation budget, **while using ≤ 1/5 of their twin CPU-seconds**.

Both halves must hold. "Matches or beats" is judged on the median normalised
gap over held-out feasible goals, with a paired bootstrap CI over goals; a tie
within the CI counts as "matches". Twin CPU-seconds are measured, not estimated,
by `TwinEvaluator`.

### H3 — verification is worth what it costs

> `surrogate_only`'s verified objective is **worse** than
> `surrogate_first_verify`'s, **and** its mean surrogate-minus-twin profit gap
> on the chosen plan is **positive** (the winner's curse).

The second clause is the mechanism, not just the outcome: picking the argmax of
a biased predictor and never checking it should over-promise. If the gap is not
positive, H3's explanation is wrong even if its first clause holds, and that
will be said.

### H4 — constraints are respected, and impossible ones are reported

> **Feasibility rate ≥ 90%** on the held-out feasible goals (point feasibility,
> on the independent re-evaluation), **and both infeasible-by-design goals
> return `status="infeasible_unverified"`**.

A method that returns a "feasible" plan for `held_infeasible_profit_floor`
(profit ≥ £2,000,000, against a landscape best of about £493,000) has a broken
constraint check, and that is a failure regardless of its gap.

---

## Verdict rule

1. **Any hypothesis that fails is reported as failed**, in the model card and in
   the summary, with the number that failed it.
2. **No goal is edited, dropped or re-weighted after held-out results are seen.**
   The goal library is frozen by this commit.
3. If a goal turns out to be **broken by a bug** — a malformed spec, an
   unreachable context, a crash — the aggregate is reported **both with and
   without it**, and the bug is described.
4. Methods that do not apply to a goal (grid above 4 free controllables) report
   **"n/a"**, never a substituted algorithm.
5. The 2 infeasible-by-design goals are **excluded from the gap and feasibility
   aggregates** and scored only on whether they return the right status.

---

## What is measured

**Primary:** normalised gap to `best_known`; feasibility rate (point and
robust).

**Cost:** surrogate evaluations, twin runs, twin CPU-seconds, wall-seconds —
all counted by the evaluators rather than estimated.

**Uncertainty:** paired bootstrap confidence intervals resampling **goals**
(not runs), because goals are the unit of generalisation. Per-goal results are
reported as well as aggregates.

**Every reported objective comes from the "reeval" seed stream**, which no
optimiser can reach. Without that, a reported value would be a maximum over the
seeds the search used.

---

## Known limitations, stated in advance

- Everything here optimises **the twin**, which is not a validated model of a
  real factory. A plan that is optimal here is optimal for a simulator whose
  `demand_capacity_ratio` is worth 45% of profit and was chosen, not measured.
- The surrogate predicts `unplanned_failures` (R² 0.733) and `downtime_hours`
  (0.907) materially worse than the other six KPIs. `held_failures_cap`
  constrains the worst of them deliberately, so if surrogate-first has a weak
  spot it should appear there.
- `best_known` is a **reference, not a true optimum**. A small gap means "close
  to the best anything found", not "close to optimal".
- The 3 optimiser seeds give a crude variance estimate. They are enough to show
  whether a difference is robust to the seed, not to put a tight interval on it.

---

## Amendment 1 — matched-budget ablation for `surrogate_only`

**Dated 2026-10-14.** Appended to the pre-registration above, which is unchanged:
the original text is byte-identical and this section is added below it. The
pre-amendment file is commit **`4f54ac8`**, blob
**`3f2690e89ed6030ffbe73eddb858437158a1fd61`**.

**Made before any held-out result existed.** At the time of this amendment the
benchmark results tree contains **zero** held-out cells — the first production
run crashed within seconds on a Sobol bug and wrote only a manifest, which was
deleted. No held-out goal has been evaluated, and no held-out result has been
looked at.

### What changed

`surrogate_only`'s TPE search budget: **20,000 → 3,000 trials**, now the same
shared constant the `sf_tpe` arm uses.

### Why

**The ablation was not matched.** `surrogate_only` exists to isolate one thing:
what twin verification is worth. It should differ from `surrogate_first_verify`
in the verification step alone — K=1 and no verification, against K=20 verified
— and in nothing else. At 20,000 trials against `sf_tpe`'s 3,000 it was also the
**best-searched** arm, so H3 would have compared "no verification but a 6.7×
larger search" against "verification with a smaller search". A result either way
would have been uninterpretable: if `surrogate_only` had won, that could have
been the extra search rather than the absence of verification.

**And it dominated the cost.** Measured at production budget on a dev goal,
`surrogate_only` was ~10,850 s per cell — **18.1 of the full run's 21.6 hours**,
for the arm that is meant to be the cheap baseline. Optuna's TPE scales at about
O(n^1.78) with the real surrogate (23.2 s at 500 trials, 57.4 s at 1,000,
174.9 s at 2,000, 599.6 s at 4,000), so 20,000 trials costs roughly 18× what
3,000 does.

**The 20,000 was not specified anywhere.** The task specification fixed budgets
for the surrogate-side random (50k), CMA-ES (20k) and TPE (3k) arms and for
twin-only (300 evaluations). It gave no budget for `surrogate_only`; 20,000 was
chosen while building Stage A, and it was the wrong choice.

### What did **not** change

- No hypothesis. H1–H4 are exactly as stated above, including H3's two clauses
  and the requirement that the surrogate-minus-twin profit gap be positive.
- No goal. The library is still the same 20 goals on the same contexts, with the
  same 2 infeasible by design.
- No threshold. The 5% gap, the 1/5 CPU-second ratio, the 90% feasibility rate
  and the AUROC gate of 0.75 are unchanged.
- **No other budget.** `sf_random` 50,000, `sf_tpe` 3,000, `sf_cmaes` 20,000,
  `twin_tpe` and `twin_cmaes` 300, `best_known` 500,000 are all unchanged.
- No metric, no normalisation, no verdict rule.

### Effect on H3

H3 becomes a cleaner test rather than an easier one. With matched search
budgets, any difference between `surrogate_only` and `surrogate_first_verify`
is attributable to verification, which is what H3 claims to measure. If
`surrogate_only` now does *better* than it would have at 20,000 trials, that
makes H3 harder to pass, not easier — a less-searched ablation is a weaker
opponent, and it has been made stronger relative to its purpose by being made
comparable.

The verdict rule above applies to this amendment too: the result is reported
whatever it is, and nothing further is changed after held-out results are seen.

---

## Amendment 2 — reduced-scale design

**Dated 2026-10-14.** Appended below the pre-registration and Amendment 1,
both of which are unchanged: the text above this section is byte-identical to
commit **`6376e45`**, blob **`18895022e8911b7b871082508193f84b3ead3ed5`** (and
the original pre-registration to `4f54ac8`, blob `3f2690e8…`).

### Why

**Runtime, on the hardware available.** The benchmark runs on an 8 GB laptop
that overheats under load. With the memory fixes, at most one `best_known` cell
at a time and at most 3 workers, the full design's 230 outstanding cells
projected to **7.8 h on a cool machine and ~14 h at the 1.8× slowdown measured
on a hot one**. The serial `best_known` lane alone (36 cells) was a 7.4 h floor
that no worker count could get under.

**Decided from timing and goal structure only.** At the time of this amendment
the held-out results tree holds 22 completed cells (listed below). **No
objective value, plan, gap, feasibility outcome or any other result field of
any held-out cell has been looked at.** The only things read from those files
were their names, their key names and their `budgets_hash`, to check that a
resume would accept them. Verdict rule 2 forbids dropping a goal *after
held-out results are seen*; none had been seen when this selection was fixed.

### The design

Recorded in `configs/benchmark_design.yaml` (design hash
**`829a2d71c70acb8b`**), not in the goal files: the goal specs, and so the goal
library hash in the results manifest, are unchanged. The design hash is in the
manifest, so a resume under a different design is refused.

**Goals — 8 of the 12 held-out goals.** The rule is structural: every goal
family stays represented; within a family the most redundant goal is moved to
a reserve; the two goals that already had completed cells are kept.

| Core (run and analysed) | Family / reason kept |
|---|---|
| `held_failures_cap` | `unplanned_failures` cap, the surrogate's weakest KPI; has cells |
| `held_infeasible_profit_floor` | infeasible by design (H4); has cells |
| `held_inventory_only` | frozen dimensions, 3 free; the only grid-applicable goal |
| `held_max_profit_service` | profit under on-time and stockout constraints |
| `held_weighted_units_cost` | weighted multi-objective |
| `held_pm_only` | frozen dimensions, 5 free (PM intervals) |
| `held_premium_supplier_service` | supplier what-if; the **only** `downtime_hours` constraint |
| `held_min_unit_cost` | minimise sense |

| Reserve (stays in the library, never run, never analysed) | Covered by |
|---|---|
| `held_max_profit` | `held_max_profit_service` (same objective, constrained) |
| `held_offshore_supplier` | `held_premium_supplier_service` (same supplier what-if) |
| `held_price_cut` | the exogenous what-if family, via the supplier goal |
| `held_speed_only` | `held_pm_only` (same 5-free frozen-dimension shape) |

**One change to the list as first proposed, made on structural grounds before
anything was run:** the proposal kept `held_offshore_supplier` and reserved
`held_premium_supplier_service`. That would have left **no** core goal
constraining `downtime_hours` — the library's coverage test requires one, and
the "Known limitations" above name `downtime_hours` (R² 0.907) as one of the
surrogate's two weak KPIs. Both are supplier what-ifs (supplier fixed to 1 or
2), so swapping them keeps the family; `held_offshore_supplier`'s only
constraint (stockout) is already covered by two other core goals. The cost is
one fewer dev/held-out mirror pair (`dev_offshore_supplier`) and no core goal
that moves the price lever (`held_price_cut`).

**The reserve is not a pool to draw from.** The reserve goals are never run
later, for any reason, including to "check" a result. The runner refuses a
reserve goal on the held-out split.

**Seeds.** `best_known`: **seed 1 only** (8 cells). Every other method: **seeds
1 and 2** (96 cells). Seed 3 is excluded by rule. Total: **104 cells**.

**Budgets: unchanged.** `best_known` is still a 500,000-candidate surrogate
sweep with the top 100 verified at 30 reps; every other budget is as in
Amendment 1.

### The 22 existing cells

**Inside the design (15), kept and used:** `held_failures_cap` × {`sf_random`,
`sf_tpe`, `sf_cmaes`, `surrogate_only`, `twin_tpe`, `twin_cmaes`} × seeds 1–2
(12); `held_infeasible_profit_floor` × `sf_cmaes` seed 1 and × `sf_random`
seeds 1–2 (3).

**Outside the design (7), kept on disk and excluded from every analysis:**
`held_failures_cap` × {`sf_random`, `sf_tpe`, `sf_cmaes`, `surrogate_only`,
`twin_tpe`, `twin_cmaes`} × seed 3 (6), and `held_infeasible_profit_floor` ×
`sf_random` × seed 3 (1). The analysis loader lists these by filename and never
opens them. No existing cell belongs to a reserve goal.

### What did **not** change

- No hypothesis. H1–H4 are exactly as stated above.
- No threshold: the 5% median gap, the 1/5 twin CPU-second ratio, the 90%
  feasibility rate and the AUROC gate of 0.75.
- No budget, no metric, no normalisation, no verdict rule, no goal spec.

### Consequences, stated before any result

- **H1** — the median normalised gap is over **7 feasible goals** (the 8 core
  goals less `held_infeasible_profit_floor`), not 11. The same 7 carry H2's
  gap comparison, H3, and H4's feasibility rate.
- **H4** still has **n = 1** for its infeasible-by-design clause: the held-out
  library only ever had one such goal (`held_infeasible_profit_floor`); the
  other is a dev goal.
- **Seed variance comes from 2 replicates**, not 3, so it can say even less
  than the "crude" estimate the Known limitations already warned about. The
  `best_known` sweep has one seed and therefore no seed variance of its own.
- **Wider confidence intervals.** The paired bootstrap resamples 7 goals
  instead of 11, and each goal's value averages 2 seeds instead of 3. A
  difference that would have been resolved at full scale may now be reported
  as a tie, and that will be said rather than read as "matches".
- Results generalise to the core families only; the price what-if is not
  tested on held-out data at all.
