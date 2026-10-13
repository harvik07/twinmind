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
