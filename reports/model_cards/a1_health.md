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

What would close it, in order of expected value:

1. **More data.** FD002 and FD004 have 260 and 249 training engines against
   FD001's 100. The published CNN-BiLSTM advantage over classical models on
   C-MAPSS is clearest on the larger, multi-regime subsets, which is where a
   sequence model's capacity starts to pay. Those runs need a GPU.
2. **Train on all 100 engines.** Much published FD001 work early-stops on the
   test set, which is leakage but also means 25% more training data. A fair
   version is k-fold cross-validated early stopping, retraining on everything.
3. **Feed the slope explicitly.** Concatenating the per-sensor linear-fit slope to
   the pooled representation would give A1 the feature the forest is winning with,
   without giving up the sequence model.
4. **Smaller trunk.** 739k parameters on 14.5k windows is a large ratio; halving
   the LSTM width is the obvious probe.

None of these were run, and none are claimed as results.

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

Given the results above, the honest position for the dissertation is that **the
random forest is currently the better RUL predictor on FD001**, and the twin
would be better served by it today. A1 earns its place only if the larger subsets
reverse that, which is the next thing to run. The Protocol makes the substitution
a one-line change, which is the point of freezing the contract first.

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
