# Model card — A1, C-MAPSS machine health

## What it is

A CNN-BiLSTM with attention pooling and two heads, predicting remaining useful
life and near-term failure probability from a 30-cycle window of turbofan sensor
readings. **A1 is the model the digital twin uses**: it supplies the per-machine
failure hazard, so nothing in the simulator depends on a hard-coded MTBF.

A1 and A2 are separate models, not two heads on one trunk. A1's input is a
`[30, 14]` sensor window; A2's is a single row of milling-machine measurements.
There is no shared input space, so there is no trunk to share. A1's *own* two
heads do share a trunk, because they share inputs — both read the same window.

## Architecture

```
input [B, 30, C]
  Conv1d(C -> 64, k=5, same) + BatchNorm + GELU + Dropout(p)
  Conv1d(64 -> 128, k=3, same) + BatchNorm + GELU + Dropout(p)
  BiLSTM(128, layers=2, batch_first, dropout=p)        -> [B, 30, 256]
  additive attention pooling over the 30 steps          -> [B, 256] + weights
  Dropout(p)
    rul_head:  Linear(256, 64) -> GELU -> Linear(64, 1)
    fail_head: Linear(256, 64) -> GELU -> Linear(64, 1)   (logit)
```

**738,626 parameters** with C = 14 sensors and p = 0.3.

Attention pooling returns its weights, which is what makes the
attention-over-cycles figure possible: it shows which positions in the window the
model actually uses, and how that shifts as an engine approaches failure.

The dropout before the heads is shared by both and stays active at inference for
MC-dropout. That placement is deliberate — sampling it perturbs the representation
that both heads read, so the RUL spread and the failure probability move together
rather than independently.

## The two heads

| Head | Target | Loss |
| --- | --- | --- |
| `rul` | Remaining useful life, piecewise-linear, capped at 125 cycles, normalised to [0, 1] by the cap | MSE |
| `fail` | Binary: will this engine fail within `h = 30` cycles? | Focal BCE, gamma 2, alpha balanced |

The failure label is **derived from the RUL target**, not from a separate source:
`fail = (rul <= h)`. That is what makes the two heads genuinely compatible — one
label, two views of it. The horizon is configurable; 30 cycles is the default.

Total loss is `mse(rul) + lambda * focal_bce(fail)`, with `lambda` selected on
validation RMSE over {0.1, 0.5, 1.0} on each run. RUL is normalised by the cap so
the two terms are commensurable; metrics are computed after un-normalising.

## Training

AdamW, learning rate **3e-4**, cosine schedule with a 5-epoch warmup, batch 256,
at most 100 epochs, early stopping on validation RMSE with patience 15, gradient
clipping at 1.0, AMP under CUDA. Best checkpoint by validation RMSE.

The learning rate is 3e-4 rather than the more usual 1e-3 because 1e-3 was
measured to overfit: the run peaked at epoch 6 of 22 and degraded steadily after.
At 3e-4 it peaks at epoch 10 of 26, and FD001 test RMSE improves 14.94 → 14.60
with the NASA score 455 → 441.

Three seeds are three **independent training runs**, not three evaluations of one
checkpoint. The log states this per run, and each writes its own checkpoint.

## Evaluation protocol

This is the part most easily got wrong, and it is the difference between an
honest number and an inflated one.

On C-MAPSS test sets, **one prediction per engine** — the RUL at that engine's
final observed cycle, compared against `RUL_FD00X.txt`. Test trajectories are
truncated before failure, and the ground truth file gives the true remaining life
at the truncation point only.

Averaging over all sliding windows of a test engine is the common way published
C-MAPSS numbers get inflated: most windows sit early in the trajectory where RUL
is pinned at the 125 cap and prediction is easy, so the mean error falls well
below the error at the point that actually matters. `evaluate_cmapss_test`
enforces the reduction, and `test_evaluate_cmapss_test_uses_only_the_last_window`
asserts that feeding it every window gives the same answer as feeding it only the
last one per unit.

The measured size of that inflation on this data is recorded in the results: the
all-window protocol reports a materially better RMSE than the last-window
protocol on the identical predictions.

## MC-dropout uncertainty

`predict_with_uncertainty(x, n_passes=30)` puts **only** `nn.Dropout` modules into
train mode. BatchNorm stays in eval throughout. Putting BatchNorm into train mode
would recompute batch statistics from the test batch — a silent train/test leak
whose symptom is a suspiciously good score. `batchnorm_layers_are_in_eval` asserts
it before sampling, and a test drives the model with a deliberately shifted batch
to prove the running statistics do not move.

**The point estimate is the deterministic pass, not the MC mean.** Averaging
stochastic passes is not the same as one clean pass: dropout perturbs the trunk
and the heads are non-linear, so `E[f(x)] != f(E[x])`. Measured on the FD001
seed-42 checkpoint, the MC mean sat **+3.58 cycles above** the deterministic
prediction, costing 0.63 RMSE (14.94 → 15.57) and 139 NASA points. The bias runs
upward — it overestimates remaining life and so schedules maintenance late, which
is exactly the direction the NASA score punishes hardest. MC sampling supplies
the standard deviation only, which is what it is for.

## Results

FD001, mean ± std over 3 independent seeds, last-window-per-unit protocol:

| Model | RMSE | NASA score | MAE |
| --- | --- | --- | --- |
| Ridge (window features) | 15.08 ± 0.08 | 374 ± 15 | 12.14 |
| **Random forest (window features)** | **12.72 ± 0.16** | **245 ± 4** | **9.51** |
| LSTM(64), single head | 14.95 ± 0.64 | 506 ± 152 | 11.05 |
| A1 CNN-BiLSTM | 15.44 ± 1.11 | 587 ± 281 | 11.36 |
| A1, 3-seed ensemble | **14.70** | **432** | 10.86 |

Per seed: 14.618, 14.707, **17.008**. The third is an outlier — one seed in
three trains badly, and its validation RMSE (18.2 against 13.2 and 14.0) shows
the failure is visible before test time rather than being test-set noise. That
instability is why the spread is ±1.11 and why the ensemble, at 14.70, beats the
per-seed mean by 0.74: averaging mostly buys robustness against the bad seed.

The live table is `reports/results/summary.md`, regenerated from
`reports/results/runs.jsonl`; the figures above are pinned here for the record.

### Targets: FD001 missed

This table is the **first** attempt, kept as the record of where the model
started. All four lines were later revisited; see "All four subsets, at the
production preset" below, where FD001 RMSE reaches 12.76 ± 0.16, FD001 NASA
reaches 265.6 ± 11.7 (target met), and FD004 is measured at 15.00 ± 0.17.

| Target | Required | Achieved | |
| --- | --- | --- | --- |
| FD001 RMSE | ≤ 13 | 15.44 ± 1.11 (ensemble 14.70) | **MISS** by 2.4 |
| FD001 NASA | < 300 | 587 ± 281 (ensemble 432) | **MISS** |
| Beat every baseline | yes | no — random forest wins | **MISS** |
| FD004 RMSE | ≤ 22 | not run on CPU | **not measured** |

**A1 does not beat the random-forest baseline on FD001, and the protocol was not
changed to make it look otherwise.**

What was tried, with measured effect on FD001 seed 42:

| Change | RMSE | Δ |
| --- | --- | --- |
| As originally specified (MC mean, lr 1e-3) | 15.86 | — |
| Deterministic point estimate | 15.26 | −0.60 |
| Learning rate 1e-3 → 3e-4 | 14.60 | −0.34 |
| 3-seed mean ensemble | 14.71 | −0.23 (from 14.94) |
| Dropout 0.2 / 0.5, weight decay 1e-3 | no consistent gain | — |

Why the gap is plausible rather than a bug: FD001 is small. After holding out 20%
of engines for validation, A1 trains on 80 engines — about 14,500 windows — for
738,626 parameters. A random forest on six hand-crafted summary statistics per
sensor (last, mean, std, linear-fit slope, min, max) has far fewer effective
degrees of freedom and is much harder to overfit at this sample size. The slope
feature in particular hands the forest the degradation trend directly, which the
sequence model must infer.

The comparison is fair: `prepare_cmapss` builds one split, and the baselines and
A1 both train on `data.x_train` from it. Neither sees the 20 validation engines.

## Remediation: seven hypotheses, and what actually moved

The gap above was diagnosed rather than tuned at. Seven hypotheses were tested
**one at a time** on FD001, each against the same baseline and the same
last-window-per-unit protocol, so every delta is attributable to a single
change. The full table is `reports/results/hypotheses.md`; the figure is
`reports/figures/a1_hypotheses_rmse.png`.

| # | change | RMSE | delta | verdict |
| --- | --- | ---: | ---: | --- |
| H0 | baseline | 14.60 | — | |
| H1 | 5-fold GroupKFold CV, single fold | 15.53 | +0.94 | hurt |
| H1 | 5-fold GroupKFold CV, fold ensemble | 14.52 | −0.08 | **no real gain** |
| H2 | smaller trunk, 738k → 100k params | 14.02 | −0.57 | **helped** |
| H3a–c | Gaussian noise alone (σ 0.01 / 0.03 / 0.05) | 14.60 / 14.60 / 14.58 | ≈0 | inert |
| H3d | noise + channel mask + jitter together | 13.24 | **−1.35** | **helped most** |
| H4a | LayerNorm in the conv block | 17.26 | +2.67 | **hurt badly** |
| H4b | no conv normalisation at all | 13.57 | −1.03 | **helped** |
| H5a | λ = 0, RUL head only | 16.11 | +1.52 | hurt |
| H5b | λ = 0.1 | 16.27 | +1.68 | hurt |
| H5c | λ = 0.5, failure head detached | 16.11 | +1.52 | hurt |
| H6 | min-max scaling to [−1, 1] | 15.44 | +0.84 | hurt |
| H7 | trend-feature branch (last, slope, delta) | 14.07 | −0.52 | helped |
| **HC** | H2 + H4b + H3d combined | **13.10 ± 0.32** | −1.50 | 3 seeds |
| **HCt** | HC + trend branch | **12.76 ± 0.16** | **−1.84** | 3 seeds |
| HCcv | HCt with a 5-fold ensemble | 13.01 | −1.59 | CV did not help here either |

Four findings worth stating plainly, because three of them contradict the
hypothesis that motivated them.

**Early stopping on a noisy validation set was not the cause (H1).** This was
the leading hypothesis: with only 20 engines held out, patience-15 early stopping
was thought to be selecting epoch 4–6 essentially at random. Five-fold GroupKFold
over engine ids, selecting on the mean validation curve, gives a **single-fold**
RMSE of 15.53 — worse than the 14.60 baseline — and a **five-fold ensemble** of
14.52, an improvement of 0.08 that is nothing against a ±1.11 seed spread.

The ensemble does gain about 1.0 over its own single folds (15.53 → 14.52), but
that is ordinary ensembling, not a fix for selection noise: each fold trains on
80% of the engines and is individually weaker. Cross-validation also costs five
training runs for that 0.08, which is a poor trade at this scale.

So the instability was real — one seed in three produced a 17.0 — but its cause
was capacity and the missing regularisation, not the early-stopping criterion.
Fixing capacity and adding augmentation dropped the spread to ±0.16 without
touching the validation protocol at all.

**The auxiliary head was helping, not hurting (H5).** The hypothesis was that the
failure head might be degrading the trunk. Removing it entirely (λ = 0) made RUL
*worse* by 1.52, as did weakening it and as did detaching its gradient. The
auxiliary task is a regulariser here, and the original λ = 0.5 was already the
right call.

**The BatchNorm fix is removal, not substitution (H4).** Replacing BatchNorm with
LayerNorm was the worst single result in the sweep, +2.67. Removing normalisation
from the conv block altogether gained 1.03. So the BatchNorm/Dropout interaction
is real, but LayerNorm is not the remedy for it in this architecture.

**Augmentation only works as a combination (H3).** Gaussian noise alone is
completely inert across two orders of magnitude of σ — 14.60, 14.60, 14.58
against a baseline of 14.60. Adding channel masking and window jitter to the same
σ = 0.03 gains 1.35. The active ingredient is the masking and jitter, not the
noise, and testing σ in isolation would have concluded that augmentation does
nothing.

**Single-seed deltas below about 1.5 are not trustworthy.** The baseline model's
seed spread is ±1.11 (v0.3-health, three seeds). The sweep runs one seed per
hypothesis for budget, so only the large effects — H3d, H4a, H4b, H5 — are
distinguishable from noise. H2's −0.57 and H7's −0.52 are suggestive rather than
established; both were re-tested inside the combination, where they held up. This
caveat is why the two combined variants were re-run across three seeds and the
verdict is taken from those means.

## Verdict: parity, and the stopping rule

The target was to beat the random forest at **12.72 ± 0.16**.

| model | FD001 RMSE | NASA | params | seed spread |
| --- | ---: | ---: | ---: | ---: |
| **Random forest** | **12.72 ± 0.16** | **245** | — | ±0.16 |
| HCt (best neural) | 12.76 ± 0.16 | 266 | 115,106 | ±0.16 |
| HC | 13.10 ± 0.32 | 311 | 100,002 | ±0.32 |
| A1 as shipped (v0.3-health) | 15.44 ± 1.11 | 587 | 738,626 | ±1.11 |

**No neural variant beat the forest on a multi-seed mean, so the stopping rule
triggered and the forest becomes the production hazard model.** HCt's best single
seed was 12.63, under the line — but its three-seed mean is 12.76, and treating
the lucky seed as the result would be the same cherry-pick this project rejects
elsewhere. The 0.04 difference is well inside either model's seed spread: these
two models are indistinguishable on this data.

### The blend beats both, and becomes the production model

Averaging the two predictions was worth more than either model alone, and by a
margin neither the sweep nor the forest tuning produced:

| model | FD001 RMSE | NASA | seed spread |
| --- | ---: | ---: | ---: |
| A1 (HCt) | 12.757 | 266 | ±0.164 |
| Random forest | 12.716 | 245 | ±0.191 |
| **50/50 blend** | **12.136** | **228** | **±0.126** |

Per seed, so this is not an averaging artefact — the blend is better than *both*
of its own components on every individual run:

| seed | A1 (HCt) | forest | blend |
| --- | ---: | ---: | ---: |
| 42 | 12.627 | 12.539 | **12.033** |
| 43 | 12.703 | 12.691 | **12.097** |
| 44 | 12.941 | 12.919 | **12.277** |

The NASA score falls furthest, 245 → 228, and that matters more than the RMSE
because the score is asymmetric: it punishes *late* predictions, which are the
ones that let an engine run past its failure point.

Why it works, and when it would stop: the two models are wrong in uncorrelated
ways. The forest sees six hand-cut summary statistics per sensor and no temporal
ordering whatsoever; A1 sees the raw ordered window through a convolutional and
recurrent trunk. They fail on different engines, so averaging cancels part of
each one's error. Two models from the same family, however well tuned, would not
give this — the gain comes from the difference in inductive bias, not from
ensembling as such.

`BlendedHealthModel` implements this behind the same `HealthModel` Protocol.
Its uncertainty follows the law of total variance for a mixture: the weighted
mean of the component variances plus the weighted spread of their means. The
second term is the useful one — when the forest and A1 disagree about an engine,
the interval widens. Measured on FD001, the blend's standard deviation correlates
+0.55 with the absolute disagreement between its components, which is the
behaviour the twin wants from a hazard it is about to sample from.

The 50/50 weight was not tuned. It is the measured optimum and also the weight a
reader would assume; fitting it per-seed on three numbers would be overfitting a
hyperparameter to the test set, which is the thing this whole document is about.

That is a more useful result than clearing the bar would have been, and it is
worth being precise about what changed:

- The gap closed from **1.88 RMSE to 0.04** — from clearly behind to statistical parity.
- The seed spread fell from **±1.11 to ±0.16**. The instability that made one seed
  in three produce a 17.0 is gone; that was the most damaging property of the
  shipped model and it was fixed by capacity reduction plus augmentation, not by
  the early-stopping change that was expected to fix it.
- Parameters fell from **738,626 to 115,106**, and training time from ~11 minutes
  to ~3.6 on the same CPU.

**The finding for the dissertation:** at C-MAPSS FD001 scale — roughly 14,500
windows drawn from 80 training engines — a tuned tree ensemble given explicit
degradation-trend features matches a carefully regularised 115k-parameter
sequence model, and comfortably beats an unregularised 738k-parameter one. The
sequence model's capacity has nothing to work with at this sample size; the
forest's inductive bias, and the slope feature handed to it directly, are worth
more than temporal modelling here. Whether that reverses on FD002/FD004, which
have 260 and 249 training engines across six operating regimes, is the open
question, and is what the GPU runs are for.

## All four subsets, at the production preset

FD002-FD004 were trained locally at the `hct` preset, three independent seeds
each, same last-window-per-unit protocol. The Kaggle GPU path could not run
them; see "Why these were trained on CPU" below.

| subset | A1 RMSE (hct) | A1 NASA score | NASA (3-seed ens.) | random forest | ridge | LSTM(64) | engines (train) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FD001 | 12.76 ± 0.16 | 265.6 ± 11.7 | **258.0** | **12.72 ± 0.16** | 15.08 | 14.95 | 100 |
| FD002 | **13.78 ± 0.17** | 908.9 ± 20.4 | 879.9 | 14.36 ± 0.16 | 16.36 | 14.96 | 260 |
| FD003 | 13.62 ± 0.31 | 644.4 ± 132.7 | 620.3 | **13.32 ± 0.49** | 15.53 | 13.83 | 100 |
| FD004 | **15.00 ± 0.17** | 1109.4 ± 54.5 | 1070.2 | 15.68 ± 0.08 | 18.51 | 14.88 | 249 |

**FD001 NASA is 265.6 ± 11.7, and 258.0 for the three-seed ensemble — below
300.** That target, missed at the time of the first write-up (587 ± 281), is
now met at the production preset. The score is the official asymmetric one:
with `d = y_pred - y_true`, `exp(-d/13) - 1` where `d < 0` and
`exp(d/10) - 1` where `d >= 0`, so late predictions are punished harder than
early ones.

The NASA scores scale far more steeply across subsets than RMSE does
(265 → 1109, a 4.2× spread, against 12.76 → 15.00, a 1.18× spread). That is the
exponential in the score doing its job: a handful of badly-late predictions on
the hard multi-regime subsets dominates the total in a way that squared error
smooths over. FD003's large seed spread (± 132.7 on a 644 mean) comes from the
same mechanism — one seed producing a few more late predictions moves the score
much more than it moves the RMSE (± 0.31).

### Checkpoint provenance

These numbers were regenerated after the checkpoint keying was made
**structural**. Checkpoint paths now carry the preset and an 8-character hash of
the architecture config (`_architecture_tag`: conv/lstm/norm/trend/detach/
dropout/window), so `a1_FD001_hct_seed42.pt` cannot be overwritten by a run at a
different preset. The previous scheme keyed on subset and seed alone, under
which `run_hypotheses.py` passed `checkpoint_dir=None` (so the HCt FD001
checkpoint was never written) and FD002's HCt checkpoint was silently
overwritten by a stray baseline cohort.

All 3 seeds × 4 subsets were regenerated and every one reloads and reproduces
its recorded RMSE to within 0.05:

```
subset      RMSE mean+/-sd  recorded   ok?       NASA mean+/-sd  ensemble
FD001     12.757 +/- 0.164     12.76   YES      265.6 +/-   11.7    258.0
FD002     13.783 +/- 0.173     13.78   YES      908.9 +/-   20.4    879.9
FD003     13.620 +/- 0.384     13.62   YES      644.4 +/-  132.7    620.3
FD004     15.001 +/- 0.174     15.00   YES     1109.4 +/-   54.5   1070.2
```

**The FD001 result does not generalise, and that is the useful finding.** The
sequence model loses on the two 100-engine subsets and wins on the two large
multi-regime ones:

- FD002 (260 engines, six operating regimes): A1 **13.78** against the forest's
  14.36 — a 0.58 win, more than three standard deviations.
- FD004 (249 engines, six regimes and two fault modes, the hardest subset):
  A1 **15.00** against 15.68 — a 0.68 win.
- FD001 and FD003 (100 engines each, one regime): forest ahead or level.

This is exactly the prediction made when the FD001 gap was first diagnosed, and
it held. At 100 engines a 115k-parameter sequence model has too little data to
beat six hand-cut summary statistics and a tree ensemble. At 250+ engines across
six regimes, where the relationship between raw sensor level and degradation
depends on which regime the engine is in, temporal modelling starts to pay and
the forest's regime-blind summary statistics stop being enough.

Targets: **FD004 ≤ 22 was the stated target and 15.00 clears it comfortably.**
FD002 and FD003 had no stated target; both come in under 14.

A caveat on the FD002 row in `reports/results/summary.md`: that file shows
15.23 for FD002, not 13.78, because a stray baseline-preset run overlapped the
production one and, being newer, won the "latest configuration" filter. Both
cohorts are real and both are in `runs.jsonl` — `132cce1d` is the 738k baseline
preset at 15.23 ± 0.49, `aa3e569a` is the production `hct` preset at 13.78 ±
0.17. The summary now carries a `config` column so the two cannot be confused.

### Why these were trained on CPU

The Kaggle GPU kernel is written, pushed and reaches the training step, but this
account's kernels run with **no accelerator and no network**: DNS resolution
fails for PyPI and for the NASA archive alike, while `kernels_pull` confirms
Kaggle stored `enable_gpu: true` and `enable_internet: true` exactly as sent.
Accepting the metadata and ignoring it is what Kaggle does for an account that
has not completed phone verification — both the accelerator and notebook
internet are gated behind it.

Verifying the phone at kaggle.com/settings and re-running
`scripts/launch_kaggle_training.py` unchanged should be sufficient. Until then
local CPU is the faster path anyway: 5–17 minutes per seed at the `hct` preset,
against a Kaggle CPU that would be slower still.

## Known failure modes

- **Late bias at high RUL.** The predicted-vs-true scatter shows most error above
  RUL ≈ 60, where the target is pinned at the cap and the sensors carry little
  signal. Errors below RUL ≈ 40 — the region that matters for scheduling
  maintenance — are visibly tighter. RMSE weights both equally, which flatters
  neither the model nor the metric.
- **57% of FD001 predictions are late** (predicted RUL above true). Late is the
  expensive direction, and the NASA score reflects it.
- **Single-regime tuning.** Everything above is FD001, which has one operating
  condition and one fault mode. FD002/FD004 span six regimes, and the
  `OperatingConditionNormaliser` path they need is implemented and tested but not
  yet exercised end to end in training.
- **Uncertainty is not calibrated.** MC-dropout std is reported (mean ≈ 5.5
  cycles) but never checked against realised error. A coverage test — do 90%
  intervals contain 90% of outcomes — is not yet written, so the band in the
  trajectory figure is indicative, not validated.

## Intended use in the twin

A1's `predict` returns a `RULPrediction` — `rul_mean`, `rul_std`,
`p_fail_within_h`, `horizon_h` — and the twin consumes exactly that through the
`HealthModel` Protocol, nothing else. The simulator turns `p_fail_within_h` into
a per-machine hazard, and `rul_std` is what lets the twin sample failure times
rather than fire them deterministically.

**The production hazard model is the 50/50 blend**, exposed as
`twinmind.models.blended_health.BlendedHealthModel` over
`ForestHealthModel` and a trained A1 (HCt configuration). It satisfies the same
`HealthModel` Protocol and returns the same validated `RULPrediction`, so the
twin's interface did not change — that is what freezing the contract before any
training code bought.

Its uncertainty does not come from MC-dropout, which it has no equivalent of.
A forest carries an ensemble spread already: every tree is an independent
estimate of the same quantity.

- `spread="tree_std"` (default) takes the standard deviation across per-tree
  predictions. On FD001 this averages 16.4 cycles.
- `spread="quantile"` reads a central interval empirically instead of assuming
  symmetry, which behaves better near the 125 cap where the distribution is
  pressed against a boundary.

On FD001 the two agree closely, because the per-tree distribution is near
symmetric away from the cap; the quantile reading earns its place on engines
predicted close to 125, where a symmetric standard deviation overstates the
upside. Both are tested.

`p_fail_within_h` comes from a **second forest fitted as a classifier** on the
same features, not from thresholding the RUL point estimate. The twin consumes
that number as a hazard, so it should be a probability the model actually fitted
rather than a deterministic function of another prediction.

A1 stays trained, reported and checkpointed as the deep-learning comparison. It
is one line away from being the hazard source again if FD002/FD004 reverse the
FD001 result.

## Reproducing

```bash
python scripts/download_data.py --dataset cmapss
```

```bash
python scripts/train_baselines.py --subset FD001
```

```bash
python scripts/train_health.py --subset FD001 --seeds 3
```

CPU wall-clock on FD001 is roughly 5–7 minutes per seed. `notebooks/kaggle_train_health.ipynb`
runs all four subsets across three seeds on a Kaggle GPU and saves checkpoints and
`runs.jsonl` as an output dataset.
