# Model C — the surrogate

A heteroscedastic residual-MLP ensemble that emulates the digital twin, so the
week-5 optimiser can screen hundreds of thousands of scenarios instead of
hundreds.

## 0. What it emulates

**The twin, not reality.** Every assumption in `configs/twin.yaml` passes
straight through: a 25% change in `demand_capacity_ratio` moves twin profit by
45%, and the 0.95 warning threshold was chosen to keep the landscape
non-degenerate rather than measured from data. **A surrogate that reproduces the
twin perfectly is still only as good as the twin**, and the twin is explicitly
not a validated model of a real factory — see
[twin_design.md](reports/model_cards/twin_design.md) §0.

So the numbers below answer "does it reproduce the simulator", and nothing more.

---

## 1. Architecture

| | |
|---|---|
| Input | 35 features — 20 scenario (18 dims, `supplier_choice` one-hot) + 15 context state |
| Trunk | 4 pre-norm residual blocks × 256, GELU, LayerNorm, dropout 0.05 |
| Heads | one per KPI → mean and softplus std with a floor of 1e-3 |
| Ensemble | 5 members: different init seeds + a bootstrap over **context blocks** |
| Loss | Gaussian NLL against the 10-rep means |
| Training | AdamW, cosine schedule with 5% warm-up, early stopping (patience 15) on held-out context blocks |
| Data | 27,840 train rows / 4,928 validation rows, **4.1 minutes on CPU** |

The bootstrap resamples **context blocks**, not rows. 64 scenarios share each
context, so resampling rows would give every member nearly the same contexts and
the ensemble disagreement would collapse to initialisation noise — understating
epistemic uncertainty exactly where it matters.

The std floor is not cosmetic: a Gaussian NLL with an unbounded-below σ has a
degenerate optimum, driving σ→0 on the easiest points and the loss to −∞.

### Target transforms

| Family | KPIs | Transform | Why |
|---|---|---|---|
| Bounded [0,1] | `oee`, `stockout_probability`, `on_time_delivery_rate` | logit, sigmoid mean head | the bound holds by construction; logit space stops the loss being squashed where the values actually sit |
| Non-negative, heavy-tailed | `units_produced`, `unit_cost`, `downtime_hours`, `unplanned_failures` | `log1p` | these span orders of magnitude; `log1p` not `log` because `unplanned_failures` is legitimately 0 |
| Signed | `profit` | standardise only | genuinely negative over much of the box (−£620k to +£493k) |

**The std transforms with the mean**, through the derivative (delta method):
σ in log space is not σ in units. Treating them as the same is the quiet error
that leaves point predictions correct and makes every calibration number
meaningless.

---

## 2. Accuracy on T1, against the noise ceiling

The test truth is a **30-replication mean of a stochastic simulator**, so it
carries noise and a perfect surrogate cannot reach R² = 1. Each KPI's ceiling is
`1 − mean(σ_rep²/30) / Var(truth)`.

| KPI | R² | ceiling | ratio | RMSE | bias | target |
|---|---:|---:|---:|---:|---:|---|
| `units_produced` | 0.9919 | 0.9990 | 0.993 | 896 | −2.7 | **HIT** |
| `oee` | 0.9905 | 0.9986 | 0.992 | 0.0253 | +0.0023 | **HIT** |
| `unit_cost` | 0.9436 | 0.9717 | 0.971 | 8.93 | −1.04 | **HIT** (by ratio) |
| `profit` | 0.9816 | 0.9961 | 0.986 | £35,370 | +£511 | **HIT** |
| `stockout_probability` | 0.9900 | 0.9984 | 0.992 | 0.0328 | +0.0011 | **HIT** |
| `downtime_hours` | 0.9070 | 0.9964 | 0.910 | 2.45 | −0.48 | **MISS** |
| `unplanned_failures` | 0.7331 | 0.9844 | 0.745 | 0.451 | −0.069 | **MISS** |
| `on_time_delivery_rate` | 0.9903 | 0.9984 | 0.992 | 0.0270 | +0.00003 | **HIT** |

**Six of eight hit. The two misses are not ceiling-limited**, and that is the
honest part. Their ceilings are 0.996 and 0.984, so the noise permits 0.95
comfortably — the model simply cannot get there.

### Why those two, diagnosed rather than guessed

Both are driven by **discrete maintenance events** rather than smooth
aggregates. `unplanned_failures` is a count with over a quarter of its mass at
exactly 0 (quantiles 0, 0, 0.47, 1.27, 2.04, max 4.33); `downtime_hours` is that
same event count multiplied by repair durations. Whether a machine fails is a
threshold comparison between a replayed trajectory's failure cycle and a PM
interval — a sharp, near-discontinuous function of the scenario, which is the
one shape a smooth MLP is worst at.

Two explanations ruled out by measurement:

- **Not extrapolation.** 0% of test values for either KPI lie beyond the
  training range.
- **Not label noise.** `unplanned_failures` has the worst noise-to-signal of the
  eight (label SE / truth sd = 0.098), and that is already in its 0.984 ceiling.

This also corrects a Stage A prediction of mine. I computed the ceilings in
advance and concluded the 0.95 target was reachable for all eight. The ceilings
*are* high enough; the model is what falls short.

### A retransformation correction, measured and not applied

`E[expm1(X)] ≠ expm1(E[X])`, so the `log1p` back-transform is biased low. The
Jensen correction `expm1(μ + σ²/2)` was measured:

| KPI | R² plain → corrected | bias plain → corrected |
|---|---|---|
| `downtime_hours` | 0.9070 → 0.9110 | −0.477 → −0.167 |
| `unplanned_failures` | 0.7331 → 0.7397 | −0.069 → −0.027 |
| `unit_cost` | 0.9436 → 0.9477 | −1.038 → −0.408 |
| `units_produced` | 0.9919 → 0.9918 | −2.71 → **+40.96** |

**Not applied.** It cuts bias 2–3× on three KPIs but makes `units_produced`
15× worse, changes no HIT/MISS verdict, and applying it to some KPIs and not
others is the kind of special-casing that confuses everyone later. Recorded as
the obvious next thing to do properly.

### Context generalisation

Every test origin is **after** every training origin. Across the 40 test origin
periods, profit R² ranges **0.9403 to 0.9934, mean 0.9807** — no collapse on the
later dates, so the surrogate is not relying on having seen a particular
forecast origin.

---

## 3. Ranking on T2, against the twin-vs-twin ceiling

1,000 scenarios at one fixed, documented context (the one nearest the median
state out of 64 candidates). The ceiling is **measured**: T2 was evaluated twice
with independent seeds, and the agreement between those two 30-rep estimates is
what a perfect surrogate would score.

| | surrogate | ceiling | target |
|---|---:|---:|---|
| Spearman ρ | **0.9917** | 0.9960 | **HIT** (≥ 0.90) |
| top-10 overlap | **8/10** | 7/10 | **HIT** (≥ 7) |
| top-50 overlap | 41/50 | 44/50 | — |
| **Regret of the argmax** | **£0** | — | — |

**The surrogate's single best scenario is the twin's single best scenario**, out
of 1,000, across a £1,134,000 profit range. That is the result that matters for
week 5: the optimiser acting on the surrogate would have picked the same plan.

The top-10 overlap of 8 **exceeds** the twin-vs-twin ceiling of 7. That is not
the surrogate beating the twin — it is a reminder that at the top of a
noisy ranking, a ±1 difference is within the noise of the yardstick itself.

---

## 4. Calibration and the epistemic signal

### Interval coverage — slightly conservative

Coverage of ±1.96 × the combined std, where the error budget includes the
**truth's own noise** (the model predicts the twin's expectation; the label is a
noisy estimate of it):

| KPI | coverage | | KPI | coverage |
|---|---:|---|---|---:|
| `units_produced` | 0.991 | | `stockout_probability` | 0.992 |
| `oee` | 0.989 | | `downtime_hours` | 0.958 |
| `unit_cost` | 0.986 | | `unplanned_failures` | 0.945 |
| `profit` | 0.991 | | `on_time_delivery_rate` | 0.988 |

Against a nominal 0.95, six of eight **over**-cover — the intervals are wider
than they need to be. Conservative is the right direction for a fallback
trigger, but it means the aleatoric head is overestimating label noise.

### The epistemic std works in-distribution and **fails at OOD detection**

| Question | Measure | Verdict |
|---|---|---|
| Does disagreement predict error? | ρ = 0.215 (profit), 0.431 mean over KPIs | weak but real |
| Does it flag a large error? | AUROC 0.736 (profit), 0.777 mean | usable |
| **Does it separate OOD from in-distribution?** | **AUROC 0.590 (profit), 0.546 mean** | **barely above chance** |

The epistemic std is only **1.19×** larger on the OOD split than on T1, and the
two distributions overlap heavily
([c_surrogate_epistemic.png](reports/figures/c_surrogate_epistemic.png), right
panel). **A 5-member deep ensemble is not a usable out-of-distribution detector
here**, and any week-5 fallback-to-twin trigger built on it would miss most
extrapolation.

The left panel shows what it *does* do: median error rises monotonically with
disagreement on both splits, and at equal disagreement the OOD errors are
higher. So it ranks risk within a split but does not calibrate across splits.

Notably, accuracy on OOD holds up better than the detector does: OOD profit R²
is **0.9238** against a ceiling of 0.9968 — degraded from 0.9816 in-distribution
but far from broken.

---

## 5. Baselines, on the identical split

Standardised RMSE in transformed space (so all eight KPIs weigh equally) and
mean R².

| Model | standardised RMSE | mean R² |
|---|---:|---:|
| Ridge | 0.5168 | 0.7321 |
| LightGBM, per KPI | 0.2780 | 0.9226 |
| Single MLP, no heteroscedastic head | 0.2745 | 0.9246 |
| **Ensemble (production)** | **0.2702** | **0.9270** |

**Did LightGBM beat the MLP ensemble? No — but only just, and not everywhere.**
The margin is 0.0044 mean R², which is small enough that it should not be read
as a decisive win. Worth recording: on the 2,048-row **dev** set the ordering was
*reversed* (LightGBM 0.8496, ensemble 0.8374). More data favours the neural
model, which is the usual shape, and it is why the comparison was repeated at
full size rather than settled early.

The ensemble ships regardless of the margin, and the reason is the epistemic std
— LightGBM gives no comparable signal, and §4 shows that signal is at least
usable for in-distribution risk ranking. **That is a trade, not a free win**: a
per-KPI LightGBM would be simpler, trains in 90 seconds, and is within noise on
accuracy.

The first version of this table was meaningless and is worth flagging: it
reported unstandardised RMSE, which is dominated by `profit` in raw pounds, so
seven of the eight KPIs contributed about a millionth of the number.

---

## 6. Learning curve

Trained on nested Sobol prefixes — a prefix of complete blocks is itself a valid
space-filling design, which is what makes this meaningful rather than an
arbitrary subsample.

| Training scenarios | profit R² | ratio to ceiling | mean R² | Spearman | top-10 |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 0.8485 | 0.852 | 0.8132 | 0.9331 | 4/10 |
| 4,096 | 0.9044 | 0.908 | 0.8601 | 0.9635 | 4/10 |
| 8,192 | 0.9468 | 0.951 | 0.8970 | 0.9815 | 6/10 |
| 16,384 | 0.9674 | 0.971 | 0.9166 | 0.9866 | 7/10 |
| **32,768** | **0.9816** | **0.985** | **0.9410** | **0.9917** | **8/10** |

**Still improving at 32,768 — it has not saturated.** Profit R² gained 0.014 in
the last doubling and the ranking gained a top-10 slot. More rollouts would buy
more accuracy; the 32,768 figure was a time-box decision, not a plateau.

---

## 7. Speed

| | |
|---|---|
| Single call | 11.9 ms |
| Batch of 10,000 | **0.079 ms per scenario** |
| Twin, 30 replications | 4.65 s |
| **Speed-up** | **59,015×** |

Target was ≥ 10,000× in batch. The single-call figure is dominated by fixed
per-call overhead across 5 members, which is why the API is batch-first: an
optimiser looping one scenario at a time would throw away 150× of this.

---

## 8. Artifacts

`models/surrogate/` — 5 member checkpoints, `scalers.npz` (feature, state and
target statistics, all from training rows only), `config.json` (architecture,
feature order, transforms). `load_surrogate(path)` returns it behind the
`SurrogateModel` Protocol.

---

## 9. Limitations

- **It emulates the twin, not reality**, and inherits every twin assumption
  including the dominant `demand_capacity_ratio` (§0).
- **Two KPIs miss the accuracy target and are not noise-limited.**
  `unplanned_failures` (R² 0.733) and `downtime_hours` (0.907) are driven by
  discrete maintenance events. Any downstream use that ranks scenarios *on
  failure count* should go to the twin.
- **The epistemic std is not an OOD detector** (AUROC 0.590). A fallback trigger
  built on it would miss most extrapolation. It is usable for ranking risk
  within a distribution, not across one.
- **Intervals are conservative**, over-covering on six of eight KPIs, so the
  aleatoric head overestimates label noise.
- **The learning curve has not saturated**; this is a time-boxed model.
- **One demand series and one product.** Context diversity comes from machine
  wear, stock and forecast origin only.
- **The `log1p` retransformation bias is uncorrected**, costing roughly 0.004
  R² and a measurable negative bias on three KPIs (§2).

---

## 10. Reproduce

```bash
python scripts/train_surrogate.py --members 5 --epochs 120
python scripts/train_surrogate.py --learning-curve
python scripts/evaluate_surrogate.py --learning-curve
python scripts/make_surrogate_figures.py
pytest tests/test_surrogate.py -v
```

Figures: [c_surrogate_profit_t1.png](reports/figures/c_surrogate_profit_t1.png),
[c_surrogate_rank_t2.png](reports/figures/c_surrogate_rank_t2.png),
[c_surrogate_learning_curve.png](reports/figures/c_surrogate_learning_curve.png),
[c_surrogate_epistemic.png](reports/figures/c_surrogate_epistemic.png).
