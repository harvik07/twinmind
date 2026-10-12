# The digital twin — design, assumptions and what is actually validated

## 0. What this is, and what it is not

**This is not a validated twin of a real factory.** No real factory data exists
for this line. There are no measured throughputs, no measured service levels and
no measured costs to compare against, and there never will be within this
project.

An earlier plan for this work promised a fidelity table comparing simulated
throughput and service level to "real held-out" values. **That table was not
built, because building it would have meant fabricating the measurements.**

The defensible claim is narrower and is the one made everywhere in this
repository:

> A simulator whose **degradation and demand processes are driven by models
> trained and validated on real industrial data**, with every assumed parameter
> declared in a register and stress-tested for how far it moves the headline
> KPIs and the ranking of scenarios.

The validity argument therefore has three limbs, and the third is the one that
matters most for any recommendation this twin ever produces:

| Limb | What it establishes | Where |
|---|---|---|
| **(a) Verification** | the code implements the intended model — analytic limits, conservation laws, determinism | [tests/test_twin.py](tests/test_twin.py), §5 |
| **(b) Component validation** | the learned parts behave, *only where real data exists* — health warnings on held-out C-MAPSS engines, the demand process against the held-out Favorita fold | §6 |
| **(c) Sensitivity** | how far the answers move when an invented constant is wrong | §9 |

**Not claimed anywhere:** validation of throughput, service level, OEE or cost,
because there is no real data for any of them.

---

## 1. Conceptual model

A five-station serial line, simulated as a discrete-event system over the **28
days following the context's forecast origin**, matching the demand horizon.

```
          (s,Q) supplier
                │
   raw yard ──► M1 ──[200]──► M2 ──[200]──► M3 ──[150]──► M4 ──[150]──► M5 ──► despatch
              cutting        forming       welding      finishing     packing
              120 u/h        110 u/h       100 u/h       115 u/h      130 u/h
                                             ▲
                                        bottleneck
```

**Flow is lot-based** (default 10 units). One SimPy event per lot rather than
per unit is what makes the performance budget reachable: a replication moves
about 12,000 lots instead of 120,000 units.

**Blocking and starving are not modelled — they emerge.** Buffers are
`simpy.Container` objects with the finite capacities above. A station's `get`
blocks when its input buffer is empty and its `put` blocks when the downstream
buffer is full. Neither is detected or special-cased anywhere. The figure
[c_twin_replication.png](reports/figures/c_twin_replication.png) shows the
result: M1 and M2 spend most of their non-processing time **blocked** behind
the welding bottleneck, while M4 and M5 **starve** downstream of it.

**Labour** is capacity-hours, not individual workers. Staffed hours per day =
`shift_count × 8 + overtime_hours`, five days in seven. A 28-day horizon is
therefore **20 staffed days and 320 staffed hours at two shifts**, not 28 and
448 — a distinction that caused a real bug (§10).

**Material** is a continuous-review (s, Q) policy on the **inventory position**
(on hand *plus* on order), consumed at M1, replenished by one of three suppliers
with different price, lead-time distribution and reliability.

**Demand** is a daily path from Model B, rescaled to plant units and adjusted
for price elasticity. Unmet demand is **lost, not backordered**.

### KPI definitions

Per replication, over the 28-day window:

| KPI | Definition |
|---|---|
| `units_produced` | finished units completed by M5 in the window |
| `oee` | availability × performance × quality over staffed hours, in [0, 1] |
| `unit_cost` | total cost ÷ units produced (total cost when nothing was produced) |
| `profit` | revenue − every cost line, tested as an identity |
| `stockout_probability` | fraction of the 28 days with any unmet demand |
| `downtime_hours` | unplanned + planned machine-hours down, summed over stations |
| `unplanned_failures` | count of failures not pre-empted by maintenance |
| `on_time_delivery_rate` | units delivered on their demand day ÷ units demanded |

OEE's **performance** term is measured against what the bottleneck could have
produced **in the uptime**, not in the planned hours. Dividing by planned hours
is the usual way this metric is reported wrongly: availability already carries
the downtime, so OEE would fall as its square. There is a test for this.

---

## 2. Health coupling — replay, not resampling

This is the part that makes it a learned twin rather than a simulator with a
neural network bolted on.

**Failures are replayed from real C-MAPSS run-to-failure trajectories.** Each
machine carries an engine id and an age in cycles, and fails when its age
reaches that engine's **actual failure cycle from the dataset**. The health
model never decides whether a machine fails.

The alternative — asking A1 for a failure probability and sampling a failure
from it — is circular. The model's belief would *be* the ground truth, so a
maintenance policy driven by the model's warnings would be evaluated against a
world the model defined, and early warning would look perfect by construction.

Instead **the health model plays the plant's monitoring system**. It sees the
sensor window at the machine's current age and issues a warning when
`p_fail_within_h` crosses a threshold. It can be early, late or wrong, and the
twin records which, because ground truth is independent of it.
[c_twin_machine_rul.png](reports/figures/c_twin_machine_rul.png) shows one
held-out engine: predicted RUL tracking true RUL down from cycle ~110, the
failure probability rising, the warning firing at cycle 221 and the real failure
at 235 — **14 cycles of notice**.

### The replay pool, and a problem the spec did not anticipate

The pool must be **out of sample**. In-sample warnings are optimistically good,
and that bias would push every scenario toward predictive maintenance: the twin
would conclude that early warning pays, because the warnings it saw were
cheating.

The repo makes this harder than expected. `prepare_cmapss` is called as
`prepare_cmapss(subset, config, seed=seed)` and passes **the training seed**
through to `split_by_unit`, so **the train/validation split is a function of the
training seed**. The three A1 seeds therefore held out three *different* engine
sets, and the intersection is nearly empty:

| Subset | Engines | Held out per seed | Held out by **all three** seeds |
|---|---|---|---|
| FD001 | 100 | 20 | **0** |
| FD002 | 260 | 52 | **3** |
| FD003 | 100 | 20 | **0** |
| FD004 | 249 | 50 | **1** |

**There is no pool that is out-of-sample for the three-seed ensemble.** The twin
therefore drives its monitoring from a **single A1 model — FD002, seed 42 — and
replays that model's own 52 held-out engines.** FD002 is also where A1 is the
production winner (13.78 against the forest's 14.36).

Two things follow, and both are disclosed rather than buried:

- The twin's warnings come from **one network, not the ensemble** the A1 model
  card reports. A real plant deploys one model, so this is not much of a
  concession, but it is a difference.
- Those 52 engines **were used for early stopping**. No gradient step was taken
  on them, but the checkpoint was *selected* by its loss on them, so they are
  not perfectly naive. This is the cleanest held-out set that exists without
  retraining, and retraining was outside the session.

Pool: **52 engines, lifetimes 143–347 cycles, mean 205.3**. A test asserts the
pool ids are disjoint from the training ids, in both the synthetic and the real
case.

### Precomputed predictions

Running A1 with 30 MC-dropout passes on the hot path is impossible — a
replication queries machine health thousands of times. `scripts/build_health_cache.py`
runs the real `HealthModel.predict_batch` **once** over every pool engine and
cycle (9,168 predictions over 52 engines) and stores the arrays flat with an
offset index. At run time the twin does an array lookup.

The cache is keyed on a hash of **the checkpoint's contents**, not its path, so
retraining under the same filename produces a different key rather than silently
reusing stale predictions — the structural-keying lesson the A1 checkpoints
learned the hard way. A data-marked test checks a cached row against a live
`predict()` call.

### Ageing and maintenance

Operating hours advance age at `cycles_per_hour = speed^wear_exponent /
life_scale_hours_per_cycle`. **Only processing time ages a machine**, not idle
time. After any maintenance the machine is **renewed with a fresh pool draw at
age 0** (as-good-as-new).

As-good-as-new is a simplification and is recorded as one. Imperfect repair
needs a repair-effectiveness parameter with nothing behind it, and the C-MAPSS
pool contains only complete run-to-failure trajectories — there is no such thing
as a half-renewed engine in the data.

Two maintenance policies, both active:

1. **Time-based PM** when operating hours since renewal reach `pm_interval_Mi`.
2. **Early warning**: when the cached `p_fail_within_h` crosses the threshold,
   PM is scheduled **at the next end of shift** — not immediately, because
   taking a machine down mid-shift on a probabilistic warning is not what a
   plant does.

`hazard_source` switches between `learned` (replayed trajectories plus
cache-driven warnings) and `fixed_mtbf` (lifetimes from a Weibull fitted to the
pool, no warnings at all). Both run; the five-scenario table in §7 reports both.

---

## 3. Common random numbers

Six independent streams — `machine_life`, `repair`, `processing`, `defects`,
`supplier`, `demand` — derived from one seed via `SeedSequence.spawn`.

This exists because the optimiser and the surrogate spend their whole lives
*comparing* scenarios. If two scenarios see differently-shuffled randomness,
their KPI difference contains both the effect of the decision and the difference
in luck, and the luck term is large: one unplanned failure moves profit by more
than most parameter changes do.

`STREAM_NAMES` is **append-only** and pinned by a test. Inserting a name in the
middle renumbers every later stream and silently invalidates every stored
result.

Replication seeds come from `SeedSequence.spawn`, never `base_seed + i`.
Consecutive integer seeds produce correlated starting states, and because the
twin spawns further children from each replication seed, neighbouring
replications can end up sharing substreams — which shrinks a confidence interval
without the information to justify it.

**Tested:** two scenarios differing only in `price_multiplier` produce an
identical failure sequence, identical PM events, and a demand path differing by
exactly the scalar elasticity factor.

---

## 4. Performance

Budget: **0.25 s per run on one core**, for ~650,000 rollouts on 8 cores in ~8
hours.

Measured over 150 runs per scenario (`scripts/benchmark_twin.py`):

| Scenario | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| baseline | 155.0 ms | 153.5 ms | 174.6 ms | 223.2 ms |
| high_overtime | **183.5 ms** | 181.3 ms | **244.0 ms** | 331.2 ms |
| high_speed | 156.0 ms | 156.0 ms | 176.7 ms | 192.4 ms |
| short_pm | 136.7 ms | 137.6 ms | 150.5 ms | 173.7 ms |
| offshore_supply | 62.7 ms | 63.6 ms | 88.5 ms | 117.5 ms |

**Budget MET.** Worst case is `high_overtime` at 183.5 ms mean / 244.0 ms p95,
both under 250 ms. That allows **1,255,634 rollouts at the mean and 944,269 at
the p95**, against a 650,000 target.

Profiling came before optimising, as required. The first working version ran at
**559 ms**, dominated by SimPy event-object allocation and per-draw numpy calls.
Three sanctioned optimisations, none of which change the model:

- **chunked pre-drawn random arrays** — `lognormal` and `binomial` were paying
  Python dispatch to produce one scalar at a time. A test asserts the chunked
  sequence is *identical* to the per-call sequence.
- **lognormal parameters computed once** per station rather than per lot.
- **cached staffed-window edge** — the staffing check runs at every lot
  boundary, ~12,000 times a replication, and was doing a binary search each time.

Result: **559 ms → 142 ms** at the baseline, with `lot_size` left at its default
of 10. No fidelity was reduced to meet the budget.

Verified under `multiprocessing` **spawn** with `Pool(2)`, which is the real
path on Windows and exactly what a lazily-built module-level cache breaks under.

---

## 5. Verification

All deterministic, all in the non-data suite, driven by a synthetic replay cache
and a stub demand model that satisfy the same contracts the real artifacts do.

| Check | Result |
|---|---|
| Determinism — same `(scenario, seed, context)` → bit-identical `KPIVector` | **pass** |
| Different seeds give different results (determinism not achieved by ignoring the seed) | **pass** |
| Flow conservation — raw in = finished + scrap + WIP | **pass**, to float precision |
| Cost identity — `profit = revenue − Σ cost lines` | **pass**, exact |
| Analytic limit — no failures, unlimited material, demand ≫ capacity → units within 1% of bottleneck × staffed hours | **pass** |
| Little's law — implied flow time inside its feasible sandwich | **pass** |
| CM cost = 5 × PM cost, and the validator rejects any other ratio | **pass** |
| CRN — price change leaves failures and demand (pre-elasticity) identical | **pass** |
| CRN — PM-interval change leaves the demand path identical | **pass** |
| CI half-width ∝ 1/√n (fitted exponent) | **pass**, exponent in (−0.75, −0.3) |
| Monotonicity under CRN — more overtime never yields fewer units, per seed | **pass**, 6 seeds |
| `KPIVector` validators hold over 1,000 random scenarios | **pass** |
| No `learned` run touches a training engine | **pass** |
| Chunked draws identical to per-call draws | **pass** |
| Runs under a spawn `Pool(2)` | **pass** |
| Every assumption flagged `assumed: true` with a reason | **pass** |

**50 non-data tests, 3 data-marked tests, all green.**

Three defects were found by these tests and fixed rather than worked around:

1. **Flow conservation was short by 9–20 units.** A station released its lot
   before its outbound `put` completed, so material sitting blocked at a full
   buffer belonged to no one. `in_process` is now released *after* the put.
2. **Demand was calibrated against a nominal 16-hour day**, overstating capacity
   by 40% — 28 calendar days hold 320 staffed hours, not 448. The baseline could
   not serve its own order book (81% stockout, −£306k profit).
3. **The scenario's (s, Q) defaults were infeasible.** See §10.

---

## 6. Component validation

Only where real data exists. Each with a stated tolerance.

### 6.1 Health warnings — **agrees**

Offline, scored on the 52 real held-out engines at the configured threshold
(0.95):

| Quantity | Value |
|---|---|
| Pool engines | 52 |
| Engines warned before failure (recall) | **1.000** |
| Warnings arriving before failure (precision) | **1.000** |
| Mean warning lead | **7.9 cycles = 11.1 operating hours** |
| **Actionable** precision (lead > the 8 h mean scheduling delay) | **0.808** |

In simulation, over 40 runs with PM intervals at their upper bound so only
warnings can intervene: **142 warnings acted on, 46 unplanned failures**, a
warning share of **0.755**.

**0.755 against 0.808, inside the ±0.15 tolerance — agrees.**

The comparison needed care, and getting it wrong first is worth recording.
Comparing the twin's warning share against offline *recall* (1.000) shows a
spurious disagreement. They are different quantities: early-warning PM waits for
the next end of shift, on average 8 hours away, so **a warning with less lead
than that is correct and useless**. The actionable precision is the like-for-like
figure, and the ~20% of failures that arrive unwarned in the twin are warnings
the plant could not act on in time — a real behaviour of the system, not an
error.

### 6.2 Fixed-MTBF Weibull — **pass**

Weibull fitted by median-rank regression to the pool's real lifetimes, compared
against them by two-sample KS.

| Quantity | Empirical pool | Fitted Weibull |
|---|---|---|
| Mean lifetime | 205.3 cycles | 205.3 cycles |
| Shape / scale | — | **5.18 / 223.3** |
| KS statistic | **0.142** (p = 0.220) | tolerance < 0.20 |

**Pass.** A shape of 5.18 is strongly increasing-hazard, which is what
run-to-failure turbofan data should look like.

### 6.3 Demand process — **corrected; the first version measured the wrong thing**

Simulated paths against the **held-out last Favorita fold**, on the twin's scale.

| Quantity | Simulated | Realised | Tolerance | Result |
|---|---:|---:|---|---|
| Mean | 974.4 | 1027.8 | ±25% | **pass** |
| SD | 200.1 | 283.4 | ±50% | **pass** |
| Residual lag-1 autocorrelation | 0.098 | 0.351 | — | **under-persistent** |
| Across-sample step dependence | **0.295** | copula target 0.330 | ±0.10 | **reproduced** |

**Band coverage, 50 series × 28 days = 1400 observations**, at a
date-aligned origin (2017-07-18, scoring 2017-07-19–2017-08-15):

| Bucket | Coverage | n |
|---|---:|---:|
| d1–7 | **0.811** | 350 |
| d8–14 | **0.746** | 350 |
| d15–28 | **0.769** | 700 |
| **overall** | **0.774** | 1400 |

Per-series spread: sd 0.109, range 0.464–0.929.

#### What the first version got wrong

An earlier version of this check reported **42.9% of realised days inside the
80% band** and that number was propagated into the limitations, the commit
messages and the pull request. **It was wrong**, for two reasons, and the second
is the kind of error that produces a confident false finding:

1. **The dates did not line up.** It drew paths from the twin's origin
   (2017-08-15, covering 2017-08-16 – 09-12) and compared them against the
   held-out fold's realised days (2017-07-19 – 08-15). Those are different days
   a month apart. The weekly phase happened to match — 28 days is exactly four
   weeks — so the result looked plausible instead of obviously broken.
2. **One series, 28 observations.** Far too few to read, as the spec's
   carry-over task said.

Aligned and measured over 1400 observations, coverage is **0.774** against a
nominal 0.80 — mildly under, not catastrophically so. It also lands within 0.004
of Model B's own out-of-sample post-conformal figure of 0.777, and reproduces
the same shape: **d8–14 is the weak bucket** here (0.746) exactly as it is in
the demand model card (0.716). Two independent measurements of the same
underlying defect agreeing is the reason to believe this one.

#### This measurement is in-sample, and cannot not be

`models/demand/config.json` records a training window of **2013-01-01 to
2017-08-15** — the entire dataset. Every day scored above is a day the
production model trained on, so **0.774 is an optimistic bound**. There is no
origin inside the data at which the saved artifact can be scored out-of-sample,
and the twin's own origin (2017-08-15) forecasts past the end of the data, where
there is nothing to score against at all.

The genuinely out-of-sample numbers exist and are quoted from training
(`reports/results/demand_folds.json`): **0.656 raw and 0.777 post-conformal**
across 200 series and 3 rolling folds, with d8–14 at 0.716. The in-sample figure
here being no better than the out-of-sample one is weak evidence that the
optimism is small — a quantile model with 7 boosters and a conformal correction
has limited capacity to memorise a particular day.

**Not claimed:** validation of throughput, service level or cost. No real data
exists for them.

---

## 7. The five illustrative scenarios

Mean ± 95% CI half-width over 30 replications, common random
numbers across scenarios (`base_seed=11`), so the differences between rows are
attributable to the decision rather than to luck.

### `hazard_source: learned` — replayed trajectories plus cache-driven warnings

| scenario | units | OEE | unit cost | profit | stockout P | downtime h | unplanned | OTD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 23334 ±661 | 0.710 ±0.019 | 27.53 ±1.12 | 332069 ±30758 | 0.211 ±0.021 | 10.3 ±0.8 | 0.40 ±0.21 | 0.817 ±0.015 |
| `throughput_+10pct` | 24504 ±797 | 0.679 ±0.021 | 26.35 ±1.20 | 337151 ±31763 | 0.204 ±0.021 | 9.5 ±0.8 | 0.37 ±0.21 | 0.823 ±0.016 |
| `throughput_+20pct` | 25930 ±667 | 0.660 ±0.016 | 24.79 ±0.74 | 349128 ±30418 | 0.189 ±0.016 | 9.2 ±0.7 | 0.40 ±0.21 | 0.833 ±0.013 |
| `short_pm_intervals` | 22516 ±671 | 0.644 ±0.017 | 29.63 ±1.22 | 285677 ±30165 | 0.243 ±0.023 | 33.1 ±1.2 | 0.00 ±0.00 | 0.799 ±0.018 |
| `high_speed` | 24864 ±700 | 0.624 ±0.017 | 26.51 ±0.81 | 322587 ±29298 | 0.205 ±0.017 | 14.6 ±1.5 | 1.83 ±0.46 | 0.824 ±0.014 |

### `hazard_source: fixed_mtbf` — Weibull lifetimes, no monitoring system

| scenario | units | OEE | unit cost | profit | stockout P | downtime h | unplanned | OTD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 23154 ±697 | 0.699 ±0.021 | 27.90 ±1.09 | 325190 ±28213 | 0.221 ±0.022 | 13.2 ±1.2 | 1.50 ±0.34 | 0.814 ±0.015 |
| `throughput_+10pct` | 24390 ±809 | 0.671 ±0.022 | 26.59 ±1.19 | 334713 ±30478 | 0.206 ±0.022 | 12.5 ±1.4 | 1.47 ±0.35 | 0.823 ±0.016 |
| `throughput_+20pct` | 25689 ±697 | 0.649 ±0.018 | 25.28 ±0.84 | 341343 ±30708 | 0.191 ±0.017 | 12.2 ±1.1 | 1.43 ±0.32 | 0.832 ±0.013 |
| `short_pm_intervals` | 22604 ±656 | 0.646 ±0.017 | 29.49 ±1.24 | 288749 ±30960 | 0.241 ±0.023 | 33.4 ±1.2 | 0.03 ±0.05 | 0.801 ±0.019 |
| `high_speed` | 24780 ±649 | 0.616 ±0.017 | 26.90 ±0.81 | 314441 ±28728 | 0.206 ±0.017 | 18.2 ±1.6 | 3.57 ±0.38 | 0.824 ±0.014 |

### Reading the table

**The monitor earns its keep on failures, and the profit claim does not yet
follow.** At the baseline, unplanned failures fall from **1.50 ± 0.34 to
0.40 ± 0.21** when the learned hazard is switched on — a 73% reduction, and the
intervals are clearly separated. Under `high_speed` the gap is wider still,
**3.57 ± 0.38 against 1.83 ± 0.46**: driving the line hard consumes life faster,
so there is more for the monitor to catch.

Profit also moves the right way (£325,190 → £332,069), **but that difference is
well inside the ±£30,758 interval and should not be read as established.** The
two modes are not cleanly paired under common random numbers either — in
`fixed_mtbf` mode a renewal draws a Weibull lifetime where the learned mode
draws an engine, so the `machine_life` stream diverges between them and the
variance reduction that makes the scenario-to-scenario comparisons tight does
not apply across the modes. Quantifying the monetary value of the learned hazard
needs a paired design built for it; that is the week-6 ablation, not this
table. What this table establishes is the failure-count effect.

**Throughput pays, and OEE falls while it does.** +20% throughput adds 2,596
units and £17,059 of profit, but OEE drops from 0.710 to 0.660. That is not a
contradiction: OEE's performance term is measured against what the bottleneck
*could* have produced, and raising the target raises the denominator faster than
the line can follow. A reader who treats OEE as a proxy for profit here will
reach the wrong conclusion, which is exactly why both are reported.

**Short PM intervals eliminate unplanned failures and destroy profit.** 48-hour
intervals drive unplanned failures to **0.00** — the strongest possible
maintenance outcome — while profit falls from £332,069 to £285,677, a 14% loss,
and stockout probability rises from 0.211 to 0.243. The maintenance is paid for
in PM events, downtime and lost production. This is the clearest trade-off in
the model and the reason the twin is worth optimising over at all.

**High speed is not free.** It adds 1,530 units but *lowers* profit by £9,482,
because `defect_speed_exponent = 3.0` turns speed into scrap and
`wear_exponent = 2.0` turns it into failures (0.40 → 1.83 unplanned). Both
exponents are assumptions (§9), so this particular conclusion is the one most
exposed to them.

---

## 8. Landscape sanity — is there a real trade-off?

If the best plans are "maximum everything", the twin has no interior trade-off,
the week-5 optimisation is trivially solved, and the whole exercise says
nothing. So before moving on: **500 scrambled-Sobol scenarios over the
configured bounds, 10 replications each**, ranked by mean profit.

| | profit |
|---|---:|
| best of 500 | £492,836 |
| median | £172,203 |
| worst | £-620,468 |

A scenario is "at a bound" in a dimension when its value is within 5% of the
dimension's range of either end. Across the **top 10** scenarios, the most
saturated dimensions are:

| dimension | top-10 scenarios at a bound |
|---|---:|
| `price_multiplier` | 2 / 10 |
| `shift_count` | 1 / 10 |
| `overtime_hours` | 1 / 10 |
| `pm_interval_M1` | 1 / 10 |
| `pm_interval_M3` | 1 / 10 |
| `pm_interval_M5` | 1 / 10 |

**Not degenerate: zero of 18 dimensions are saturated** (the threshold is 9 of
10). The most any dimension manages is `price_multiplier` at 2 of 10 — barely
above what random placement would give. The best plans sit in the **interior**
of the decision space in every dimension.

Profit spans **£-620,468 to £492,836** across the
sample, so the decisions matter enormously while none of them is a simple
"turn it up" lever. That is the condition the week-5 optimiser needs.

This did not hold on the first attempt, and the fix is recorded rather than
quietly applied: at a warning threshold of 0.5 the monitor caught every failure,
`unplanned_failures` was identically zero and the maintenance dimensions were
inert. See §10.3.

---

## 9. Assumptions, sensitivity and rank stability

### 9.1 The register

Every parameter with no data behind it is declared in `configs/twin.yaml` under
`assumptions:`, each flagged `assumed: true` with a reason. A config validator
**rejects an entry with `assumed: false`** — a data-backed parameter does not
belong in the register, and an assumed one must not be able to hide by omitting
the flag. The simulator reads them only through `TwinConfig.assumption(name)`,
which raises on an unknown key, so every number the twin treats as assumed is
one the sensitivity study can find by enumerating the register.

### 9.2 One-at-a-time, ±25%, with common random numbers

30 replications per point at the baseline scenario. Rank stability is the
Spearman correlation between the profit ranking of 50 Sobol scenarios
under the nominal assumption and under the same assumption raised 25%, at
5 replications each.

| assumption | nominal | profit at −25% | profit at +25% | rank ρ | ranking stable? |
|---|---:|---:|---:|---:|---|
| `demand_capacity_ratio` | 0.85 | -45.42% | -2.98% | 0.881 | yes |
| `lead_time_multiplier` | 1 | +20.51% | -25.62% | 0.955 | yes |
| `demand_spread_multiplier` | 1 | +2.68% | -3.20% | 1.000 | yes |
| `defect_base_probability` | 0.012 | +2.80% | -2.32% | 0.999 | yes |
| `life_scale_hours_per_cycle` | 1.4 | -1.82% | +0.54% | 0.998 | yes |
| `processing_cv` | 0.25 | +0.01% | +0.30% | 0.999 | yes |
| `unplanned_repair_multiplier` | 1 | +0.07% | -0.07% | 1.000 | yes |
| `wear_exponent` | 2 | +0.00% | +0.00% | 1.000 | yes |
| `price_elasticity` | -1.2 | +0.00% | +0.00% | 0.991 | yes |

**Every assumption is rank-stable (ρ ≥ 0.881).** No scenario ranking in this twin
depends on getting an invented constant right to within 25%, which is the result
that matters most: a recommendation produced here survives its assumptions being
wrong by a quarter, even where the absolute KPIs move a great deal.

**Demand level dominates, and it is an assumption.** `demand_capacity_ratio`
costs **−45.4% of profit** when lowered 25%, far more than anything else, and
carries the weakest rank correlation in the table (**0.881**). The asymmetry is
the fixed-cost structure: overhead is £4,200/day and labour is paid by the
staffed hour whether or not there is an order book, so demand falling 25% takes
revenue with it while most costs stay. Every absolute profit figure in this card
is contingent on a number chosen to make the line roughly 85% loaded, not
measured. It is also the assumption to watch as the optimiser gets to work,
since 0.881 is the closest anything comes to the 0.8 line.

**Supplier lead time is second and is the one worth measuring for real.** A 25%
longer lead time costs **−25.6% of profit**. The line is material-constrained
rather than capacity-constrained, and unlike the demand ratio this is a quantity
a purchasing department would actually know.

**A null result that was a harness bug, not a finding.** `demand_capacity_ratio`
first measured **exactly 0.00%**, which read as "the demand level does not
matter" — the opposite of the truth. It is consumed once in `build_resources` to
derive `demand_scale`, and the simulation never reads it again, so perturbing it
after the resources were built changed nothing at all. The study now recalibrates
`demand_scale` alongside it, which is what turned a 0.00% into the largest effect
in the table. A sensitivity study that silently fails to propagate a parameter is
worse than no study, because it certifies the parameter as safe.

**Two parameters are exactly inert at the baseline, and correctly so.**
`wear_exponent` enters as `speed ** wear_exponent` and `price_elasticity` as
`price_multiplier ** elasticity`. At the baseline both bases are exactly 1.0, so
any exponent gives 1.0 and the measured effect is identically zero — correct
arithmetic and useless evidence. Re-measured at a scenario where they can act
(speed 1.20, price 1.15, reference profit £324,137):

| `wear_exponent` | +0.61% | -0.31% |
| `price_elasticity` | +7.79% | -8.09% |

`price_elasticity` is worth ±8% of profit once price moves off 1.0, so the
baseline-only design would have badly understated it. `wear_exponent` stays
small even off-baseline. This is a limitation of one-at-a-time-at-a-point
designs generally: a parameter can be inert at the point chosen and important
everywhere else.

### 9.3 Demand spread, and the bias it inherits

`demand_spread_multiplier` rescales each sampled demand path's deviation from
its own median, inside the twin, after `sample_paths` and never inside the
frozen `DemandModel` Protocol. It exists because Model B's post-conformal
coverage is uneven — d8–14 averages **0.716** against a [0.76, 0.84] target, and
on fold 1 an 80% interval covered **65%** of outcomes — which §6.3 confirms
inside the twin at **0.746** for d8–14 against 0.774 overall. The paths are
mildly too narrow at mid-horizon; the question is what that does to the answers.

Perturbed over {0.8, 1.0, 1.2, 1.4} at the baseline, 30 replications each:

| multiplier | stockout P | Δ vs 1.0 | OTD | Δ vs 1.0 | profit | rank ρ |
|---|---:|---:|---:|---:|---:|---:|
| 0.8 | 0.2506 | +0.0049 | 0.8001 | +0.0003 | 316585 | 0.999 |
| 1.0 | 0.2457 | +0.0000 | 0.7998 | +0.0000 | 309765 | 1.000 |
| 1.2 | 0.2420 | -0.0037 | 0.7988 | -0.0009 | 301908 | 1.000 |
| 1.4 | 0.2407 | -0.0049 | 0.7976 | -0.0021 | 294033 | 0.999 |

**By the stated test, neither movement is material.** Maximum absolute change in
stockout probability is **0.0049** and in on-time
delivery **0.0021**, both below the 0.01 threshold, and the
minimum rank correlation is **0.999**. Scenario rankings are
untouched.

**But the direction flips with how well-provisioned the line is, and that is the
part worth stating.** At the baseline, widening the spread *reduces* stockout
probability (0.2506 → 0.2407). At the best scenario the landscape search found —
stockout 0.038, OTD 0.980 — it *raises* it:

| multiplier | baseline stockout | landscape-best stockout |
|---|---:|---:|
| 0.8 | 0.2506 | 0.0370 |
| 1.0 | 0.2457 | 0.0383 |
| 1.2 | 0.2420 | 0.0383 |
| 1.4 | 0.2407 | 0.0395 |

The mechanism is the metric. `stockout_probability` counts **days with any unmet
demand**, not how much went unmet. At the baseline the line is short on most
binding days already, so widening the spread mostly adds low days that *can* be
served and removes them from the count. On a well-provisioned line the binding
event is a rare high day, and widening creates more of them.

**So the honest statement of the bias.** On the well-provisioned plans an
optimiser would actually recommend, the twin's too-narrow demand means stockout
risk is **understated**, in the direction the demand model card predicted — but
only by about **+0.0012 absolute** per 20% of spread, which is negligible against
the 0.05 constraint. On a badly-provisioned plan the sign reverses. **Any
stockout-risk number from this twin inherits Model B's known mid-horizon
undercoverage, and the sign of that inheritance depends on the scenario**, so it
cannot be corrected with a single offset.

**Profit is the KPI that does move**, monotonically and in one direction: a 40%
wider spread costs about **5% of profit at both the baseline (£316.6k → £294.0k)
and the best scenario (£477.3k → £452.3k)**. That is material for any absolute
profit claim, though not for ranking. It is driven by the stockout penalty on
the magnitude of shortfall, which — unlike the day-count — does rise with spread
at every provisioning level.

---

## 10. Where the spec was wrong about this repo or this data

Reported as asked, with what was done instead.

1. **"Use the validation-split engines"** assumes one validation split. There is
   one *per training seed*, because `prepare_cmapss` passes the training seed to
   `split_by_unit`. No pool is out-of-sample for the three-seed ensemble (FD002
   intersection: 3 engines; FD001 and FD003: 0). **Resolved** by driving the
   monitor from a single model, FD002 seed 42, with its own 52 held-out engines.

2. **"Mean machine life ≈ 40 operating days, so failures occur within a 28-day
   window"** cannot both hold. Forty two-shift days is 640 operating hours; a
   28-day window contains only **320 staffed hours**, and a station processes
   for less than that. Measured at 40 days: **0 failures and 0 PM events** in the
   baseline — every maintenance lever inert. **Resolved** by setting
   `mean_life_operating_days` to **18**, which is where failures actually occur
   inside the horizon.

3. **A warning threshold of 0.5 makes `unplanned_failures` identically zero.**
   The 30-cycle failure horizon gives ~42 processing hours of notice against a
   ≤16 hour scheduling delay, so a sensitive monitor catches everything and the
   KPI is degenerate. Measured over 8 seeds with PM at its upper bound:

   | threshold | unplanned failures | PM events | mean lead |
   |---|---:|---:|---:|
   | 0.50 | 0.00 | 4.12 | 56.0 h |
   | 0.80 | 0.00 | 3.50 | 22.9 h |
   | 0.90 | 0.00 | 3.12 | 12.5 h |
   | **0.95** | **0.62** | 2.50 | 6.3 h |
   | 0.99 | 2.88 | 0.12 | 1.8 h |

   **Resolved** by setting it to **0.95**, where the monitor is informative but
   imperfect. This is the spec's own "fix the structure and say so" step, applied
   to a policy parameter; it was chosen for a non-degenerate landscape, not
   measured from data.

4. **The frozen `Scenario` inventory defaults were infeasible.** With (s, Q)
   reviewed on inventory *position*, an order quantity **above** the reorder
   point means one delivery lifts the position past the trigger, so a second
   order can never be in flight. Supply is capped at Q per lead time: **357
   units/day** at Q = 2500 with a 7-day lead, against ~970 demanded. **Resolved**
   by changing the defaults to 7000 / 6000 / 1500. The 18-D contract is
   untouched — dimensions, names and order are unchanged.

5. **`twin.yaml` had `horizon_days: 90`** while the demand contract and the
   spec both call for 28 — and the simulator used 28 regardless, so the config
   said one thing and the code did another. **Resolved** by setting it to 28.
   `warmup_days` was 7 and discarded nothing; it is now 0, with the absence of a
   warm-up discard recorded as a limitation (§11) rather than implied by a
   config key that did nothing.

---

## 11. Limitations

- **This is not validated against a real factory, and cannot be.** Throughput,
  service level, OEE and cost have no measured counterpart. Only the health and
  demand components are validated, and only against the data they came from.
- **The twin mildly understates stockout risk at mid-horizon.** Band coverage is
  0.774 overall against a nominal 0.80, with d8–14 at 0.746 — outside the
  [0.76, 0.84] target band and matching Model B's known defect. The effect on
  stockout probability is small and sign-dependent; see §9.3. An earlier version
  of this card reported 42.9% coverage from a date-misaligned single-series
  measurement; that figure was wrong and is corrected in §6.3.
- **Demand paths are under-persistent.** Residual lag-1 of 0.098 against a
  realised 0.351. Real demand clusters more than the simulation does, which
  again biases stockout risk downward, because clustered shortfalls are what
  empty a finished-goods store.
- **The monitor is a single network, not the ensemble**, and its pool was used
  for early stopping (§2).
- **As-good-as-new repair.** No imperfect-repair model, because there is no data
  for one and the pool contains only complete trajectories.
- **One demand series drives the whole line.** The twin makes one product.
- **The 28-day horizon has no warm-up discard.** Contexts start machines
  part-worn and the yard stocked, so the opening transient is short, but it is
  not removed.
- **Several parameters are chosen for a non-degenerate landscape**, not measured
  — most importantly the warning threshold (§10.3). Any claim about the value of
  predictive maintenance inherits that choice.

---

## 12. Reproduce

```bash
python scripts/build_health_cache.py
python scripts/benchmark_twin.py --runs 150
python scripts/validate_twin.py --all
python scripts/make_twin_figures.py
pytest tests/test_twin.py -v
```

Figures: [c_twin_machine_rul.png](reports/figures/c_twin_machine_rul.png),
[c_twin_replication.png](reports/figures/c_twin_replication.png),
[c_twin_ci_scaling.png](reports/figures/c_twin_ci_scaling.png),
[c_twin_sensitivity_tornado.png](reports/figures/c_twin_sensitivity_tornado.png).
