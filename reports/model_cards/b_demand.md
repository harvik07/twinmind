# Model B — Demand (Favorita)

Produces a 28-day-ahead probabilistic demand forecast for the digital twin: a
q10/q50/q90 band per day, and correlated sample paths for the SimPy inventory
simulation to consume.

- **Branch / tag:** `feat/model-demand` → `v0.4-demand`
- **Data:** Corporación Favorita grocery sales, top 200 series by volume
  (75.5% of total units across 1,782 store×family series)
- **Horizon:** 28 days, direct multi-horizon
- **Protocol:** 3 chronological rolling-origin folds, identical for every model
- **Production artefact:** `models/demand/` (7 LightGBM boosters in `.txt`,
  `copula.npz`, `conformal.npz`, `config.json`)

---

## 1. Which model won, and by how much

All models are scored on the **same three folds, the same rows, the same
indices**, inside one loop in `scripts/train_demand.py`. Values are the mean ±
sd across folds (n_folds = 3).

| Model | RMSLE | MAE / best-naive MAE | q50 pinball |
|---|---|---|---|
| Seasonal naive, lag 7 | — | 1.000 (reference) | — |
| Seasonal naive, lag 364 | — | — | — |
| LightGBM point (L2) | 0.2950 ± 0.1770 | — | 229.23 ± 47.33 |
| **LightGBM quantile (production)** | **0.2527 ± 0.1020** | **0.8784 ± 0.1579** | **221.26 ± 39.29** |
| N-HiTS (MQLoss) | see §6 | see §6 | see §6 |

**The LightGBM quantile model is the production model.** It beats the L2 point
booster on both RMSLE (0.2527 vs 0.2950, a 14.3% reduction) and on q50 pinball
(221.26 vs 229.23, 3.5% better).

That second comparison is worth a sentence, because the first version of it was
meaningless. The "point" baseline was originally a LightGBM fitted at
`alpha=0.5` under the quantile objective — which *is* the quantile model's own
median booster, so the table was comparing the model against itself and the
margin was noise. The point model is now fitted with `objective="regression"`,
genuinely estimating the conditional mean under squared error. That is a
different estimator, and the comparison is real.

Against the seasonal naive baselines:

| Metric | Value | Zero denominators excluded |
|---|---|---|
| MASE vs lag-7 | 0.9953 ± 0.1195 | 0 |
| MASE vs lag-364 | 0.5238 ± 0.0560 | 0 |
| Relative MAE vs **best** naive | 0.8784 ± 0.1579 | — |

Both seasonal-naive denominators are computed on **training data only** and
zero denominators are excluded and counted; the count was zero in every fold,
so nothing was silently dropped.

MASE vs lag-7 is 0.9953. The learned model barely beats last week. This is the
honest headline and it should not be buried: on a top-200 high-volume retail
subset with strong weekly structure, a one-line seasonal naive is a genuinely
hard baseline, and 28 days out the model's edge over it is about half a
percent. The lag-364 comparison looks much better (0.52) only because the
year-ago baseline is much worse. Relative MAE is reported against the *better*
of the two naives, which is the strict version of the comparison.

### Targets

| Target | Result | Value |
|---|---|---|
| Relative MAE ≤ 0.85 | **MISS** | 0.8784 |
| 80% coverage in [0.76, 0.84] | **HIT** (post-conformal) | 0.7765 |
| Quantile beats point on q50 pinball | **HIT** | 221.26 vs 229.23 |

Two of three. The relative-MAE miss is not marginal enough to explain away:
0.878 against a 0.85 target, with a fold sd of 0.158. It is reported as a miss.

---

## 2. Leakage strategy

The forecast is **direct multi-horizon with a horizon-step feature**. One
training row is `(series, origin, k)` for `k = 1..28`. Every history feature is
computed **as of the origin** and is therefore identical across all 28 rows of
that origin; `k` tells the model how far ahead it is predicting.

This exists to avoid a specific trap. `twinmind.data.features` builds lags
{1, 7, 14, 28} shifted by one day, which is correct for one-step-ahead
forecasting and silently wrong for 28. At a 28-day horizon, `lag_1` for target
day `origin+28` is the sales of day `origin+27` — a number that does not exist
when the forecast is made. A model trained on that column validates beautifully
and is unusable in the twin.

Every covariate is classified, in `src/twinmind/models/demand_features.py`:

| Class | Members | Read from |
|---|---|---|
| **Known future** | `dow`, `dom`, `month`, `week_of_year`, `is_weekend`, `is_payday`, the 5 holiday flags, 6 weekly + 10 yearly Fourier terms, `onpromotion` | the **target** day |
| **Static** | `store_nbr`, `cluster` | constant |
| **Origin only** | `oil_price`, `transactions` | the **origin** day (`*_at_origin`) |
| **Target history** | lags 1/7/14/28/364, rolling mean+sd over 7/28/91, 28-day trend, 28-day nonzero rate | strictly ≤ origin |

`onpromotion` is classified as known-future on the stated assumption that
promotions are *planned*, so a retailer does know them in advance. In the twin
this becomes a scenario lever, which is the point of the assumption.

`transactions` is classified even though the current featurised frame does not
carry it. It is one of the six Favorita tables and an obvious later merge, and
store footfall is emphatically not known 28 days ahead. Classifying it now
means adding the column cannot silently become a leak.

The target is modelled under `log1p`. Quantiles are equivariant under monotone
transforms, so the q10/q50/q90 of `log1p(y)` map back through `expm1` to the
q10/q50/q90 of `y` exactly — no smearing correction is needed, which would not
be true of a mean forecast.

### The guard, and its negative control

`origin_features(history, target)` reads only the frame it is handed, and the
caller slices that frame to end at the origin. Truncation invariance therefore
holds by construction rather than by care — but "by construction" is a claim
that needs a test that can fail.

`DemandData.truncate(origin)` blanks the **unknowable** columns on future rows
(realised sales, oil price, transactions) and keeps calendar, holidays and
planned promotions. An earlier version deleted future rows outright and the
invariance test failed on 100% of elements — correctly, because deleting the
known-future covariates made the model fall back to zeros for `onpromotion` and
`is_holiday`, i.e. the test was measuring the removal of information a deployed
system genuinely has. `hard=True` still drops future rows entirely, for callers
that want that.

Two tests in `tests/test_demand.py`:

- `test_forecast_is_invariant_to_truncation` — predictions from the full frame
  and from the truncated frame are elementwise identical.
- `test_negative_control_leaky_feature_breaks_truncation_invariance` — a
  deliberately leaky builder (`_leaky_inference_rows`, which adds a `lag_1`
  shorter than the step ahead) is run through **the same guard**, and the test
  asserts the guard **fails** on it.

Evidence from the final run:

```
REAL builder  PASSES (no leak)
LEAKY control FAILS -> leaking via ['leaky_lag_1']
```

A guard that has never been seen to fail is not evidence of anything.

---

## 3. The conformal step

Raw 80% interval coverage on the full 200-series run was **0.6562 ± 0.0155** —
well below nominal, so split-conformal calibration is applied.

By horizon bucket, before calibration:

| Bucket | Coverage |
|---|---|
| d1–7 | 0.700 |
| d8–14 | 0.630 |
| d15–28 | 0.647 |
| overall | 0.656 |

The shortfall is real but not pathological. A systematically over-confident
quantile booster is the expected failure mode here: LightGBM's pinball
objective is fitted per-quantile with no mechanism tying the two tails to an
actual interval, and the direct multi-horizon setup means a single origin's 28
rows share all their history features, so their errors are strongly dependent
and the effective sample size behind each quantile is far smaller than the row
count suggests. 0.656 against 0.80 is consistent with that; it is not the
near-zero coverage that would indicate a broken transform or a misaligned join.
No deeper cause was pursued, on the stated threshold that below ~0.6 would have
warranted it.

Calibration is **CQR-style split conformal with per-step offsets**: one offset
per horizon step `k`, from the conformity scores
`max(q10 − y, y − q90)`, at the finite-sample-corrected level
`ceil((n+1)(1−α))/n`. Per-step rather than pooled, because the width needed at
day 1 and day 28 are not the same quantity.

After calibration: **0.7765** overall (d1–7 0.830, d8–14 0.716, d15–28 0.780).
Inside the [0.76, 0.84] target band.

---

## 4. Fold fairness — what was calibrated on what

`make_rolling_origin_splits(n_splits=3, horizon=28)`:

| Fold | Train through | Validate |
|---|---|---|
| 0 | 2017-05-23 | 2017-05-24 … 2017-06-20 |
| 1 | 2017-06-20 | 2017-06-21 … 2017-07-18 |
| 2 | 2017-07-18 | 2017-07-19 … 2017-08-15 |

Chronology is asserted in `_fold_frames`, not assumed.

- **Conformal offsets are fitted on fold 0 only.**
- **Conformal-adjusted coverage is scored on folds 1 and 2 only** — the folds
  the offsets have never seen. Reported as `coverage_after = 0.7765`.
- **Every other metric** (RMSLE, MAE, MASE, pinball, relative MAE,
  pre-conformal coverage) is averaged over **all 3 folds**.
- **Every model** — both seasonal naives, the L2 point booster and the quantile
  booster — is scored inside the same `_fold_frames` loop on **identical
  evaluation indices**. No model sees a fold another model does not.

`n_folds = 3` is recorded in the results table. The honest caveat: with three
folds, a fold standard deviation is an estimate from three numbers, and every
± in this card should be read as indicative rather than as a confidence
interval.

---

## 5. Sampling: Gaussian copula over calibrated marginals

The twin needs *paths*, not bands. Drawing each day independently from its own
marginal would produce a trajectory that zig-zags across the band every day and
would badly understate the probability of a sustained run of high demand —
which is exactly the event an inventory policy must survive.

`sample_paths` therefore:

1. Predicts the 7-quantile grid (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95) for all
   28 steps via `_predict_grid`.
2. **Applies the conformal offsets**, then `np.sort(..., axis=1)` to restore
   monotonicity. This is the single shared code path — `quantiles()` and
   `sample_paths()` both call `_predict_grid`, so the sampler's marginals are
   the *calibrated* ones by construction, not the raw boosters'.
3. Draws `z ~ N(0, Σ)` and maps through the normal CDF to uniforms.
4. Inverts each day's empirical quantile grid at those uniforms
   (`interpolate_quantiles`, flat beyond the 5th/95th).

`Σ` is fitted in `fit_copula` from the normal scores of training residuals,
shrunk 0.3 toward an AR(1) target, and repaired to the nearest PSD matrix by
eigenvalue clipping (`nearest_psd`) so the Cholesky cannot fail on a
near-singular estimate.

### Checks

**Marginal fidelity** — `test_sampled_marginals_match_the_calibrated_quantiles`
draws paths and compares their empirical quantiles to the calibrated grid.

This test initially failed at 17.7% of interval width, which looked like a bug
and was not: it was Monte Carlo noise in the quantile estimate. Measured error
fell 17.7% → 4.7% → 2.5% at n = 5k / 20k / 80k draws, i.e. the 1/√n signature
of estimation error, not bias. The tolerance is now
`max(0.05 × width, 3 × MC_SE)` with the standard error estimated from
between-batch spread, so the test is tight where it can be and does not fail on
sampling noise.

`test_sampler_marginals_follow_the_conformal_adjustment` separately asserts
that changing the conformal offsets moves the sampled marginals — i.e. that
step 2 above is actually wired in.

**Autocorrelation** — `test_copula_reproduces_the_reference_autocorrelation`
checks that the lag-1..lag-7 autocorrelation of sampled paths matches the
reference structure Σ was fitted to. (This test first reported `lag-1 nan`,
caused by a fixture that initialised step 0 to all zeros — no variance, so no
correlation. The fixture now seeds from the stationary distribution.)

**Non-crossing** — enforced by the sort in `_predict_grid` and asserted by the
`DemandQuantiles` validators, so a crossed band cannot leave the contract.

---

## 6. The one neural comparison

<!-- NHITS_RESULTS -->

---

## 7. Not comparable to the Kaggle leaderboard

**The RMSLE above must not be read against the Favorita competition
leaderboard.** They are different numbers that happen to share a name:

- This evaluates **200 series**, chosen as the highest-volume store×family
  pairs; the competition scores **all 1,782**. High-volume series are the
  easier ones — denser, less intermittent, stronger weekly structure.
- This uses **3 rolling-origin folds inside the training period** ending
  2017-08-15; the competition scores a **single fixed 16-day window** on a
  held-out future the entrants never saw.
- The competition horizon is **16 days**; this is **28**.
- The competition uses a **weighted** NWRMSLE with per-series weights; this is
  unweighted.

Any of those four differences alone would break comparability. The number is
meaningful only against the baselines in §1, which were computed under
identical conditions.

---

## 8. Limitations

- **The margin over seasonal naive is thin.** MASE vs lag-7 of 0.9953 means the
  learned model is worth about half a percent over "same weekday last week" at
  this horizon on this subset. The relative-MAE target was missed. A reviewer
  should treat the value of Model B as "produces a calibrated distribution and
  correlated paths", not "forecasts much better than a trivial rule".
- **Coverage needed conformal to reach nominal.** The raw boosters are
  over-confident (0.656 vs 0.800). The fix is sound and held out, but it is a
  correction applied after the fact, and its offsets were fitted on a single
  fold of 28 days.
- **Conformal offsets are stationary.** They are fitted once and applied
  unchanged to every future origin. If volatility shifts — a promotion regime
  change, a new store — the intervals will be wrong in the direction of the
  shift until refitted.
- **`onpromotion` is assumed known.** If promotions are in fact decided inside
  the forecast horizon, this model has been given information it would not have.
  The assumption is deliberate and becomes a twin lever, but it is an
  assumption.
- **The copula is a single global Σ.** One correlation structure is shared
  across all 200 series. Series with genuinely different persistence get the
  average one. A per-cluster Σ is the obvious next step and was not done.
- **Gaussian dependence has no tail dependence.** A Gaussian copula drives
  joint extremes to independence in the tails: simultaneous spikes across many
  days will be under-sampled relative to reality. For inventory stress-testing,
  this biases toward optimism — which is the wrong direction for a safety-stock
  decision, and is the most important caveat in this list.
- **200 of 1,782 series.** Nothing here speaks to the intermittent long tail,
  where a zero-inflated model would likely be needed instead.
- **3 folds, ~84 validation days total.** Every ± in this card rests on three
  numbers.

---

## 9. Reproduce

```bash
python scripts/train_demand.py --top-n 200 --folds 3
python scripts/train_demand_nhits.py --top-n 200 --max-steps 300
python scripts/make_demand_figures.py
pytest tests/test_demand.py -v
```

Figures: `reports/figures/b_demand_fan_chart.png`,
`b_demand_coverage_calibration.png`, `b_demand_pinball_by_model.png`,
`b_demand_sampled_paths.png`.
