# Data card — AI4I 2020 Predictive Maintenance

## Source and licence

The AI4I 2020 Predictive Maintenance Dataset, contributed by Stephan Matzka
(Hochschule für Technik und Wirtschaft Berlin) to the UCI Machine Learning
Repository as dataset 601. TwinMind fetches it through the `ucimlrepo` client,
falling back to the UCI archive URL if that fails.

It is released under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0), which permits commercial and academic reuse with attribution. The
originating publication is S. Matzka, "Explainable Artificial Intelligence for
Predictive Maintenance Applications", *Third International Conference on
Artificial Intelligence for Industries*, 2020.

The data is **synthetic**. It was generated to reflect the behaviour of a real
milling machine, but no row corresponds to an actual observed failure. That makes
it well suited to its role here — a clean, physically-plausible training signal
for a failure-probability head — while ruling out claims about real-world
failure rates.

## What it contains

Ten thousand rows, each a snapshot of one machining operation. Alongside an
identifier and a product-quality variant, there are five process measurements:
air temperature (K), process temperature (K), rotational speed (rpm), torque (Nm)
and tool wear (min). Product quality is L, M or H in a 60/30/10 split — 6,000
low, 2,997 medium and 1,003 high in the actual file — and the quality variant
shifts the failure thresholds, so it is genuinely predictive rather than a nuisance
column.

After loading, TwinMind returns eight feature columns: the five measurements plus
a three-way one-hot encoding of product type.

## Target definition

The binary column `machine_failure` indicates whether the machining operation
ended in a failure of any kind. **339 of the 10,000 rows are failures, a positive
rate of 3.39%.**

That imbalance is the dominant modelling consideration. Accuracy is a useless
metric here — always predicting "no failure" scores 96.6% — so Model A's failure
head is trained with a weighted binary cross-entropy and evaluated on precision,
recall and area under the precision-recall curve rather than on accuracy or ROC-AUC.

## The leakage trap

This is the single most important thing to know about this dataset, and it is why
`load_ai4i` returns three objects rather than one frame.

Five further binary columns record the *individual failure modes*: `TWF` (tool
wear failure), `HDF` (heat dissipation failure), `PWF` (power failure), `OSF`
(overstrain failure) and `RNF` (random failure). They are not auxiliary features.
They are the components the target is assembled from, and their logical OR
reconstructs it almost exactly.

Measured on the actual file:

- `OR(TWF, HDF, PWF, OSF, RNF) == machine_failure` for **99.73%** of rows.
- Excluding the random mode, `OR(TWF, HDF, PWF, OSF) == machine_failure` for
  **99.91%** of rows.

The residual disagreement is almost entirely RNF, the "random failure" mode, which
fires on 19 rows and is deliberately decoupled from the target on most of them.

A classifier handed those five columns will report an AUC indistinguishable from
1.0. It has learned nothing about tool wear, torque or heat dissipation; it has
learned to read the answer key. Because the score looks spectacular, this failure
mode does not announce itself — it looks like success.

TwinMind's countermeasures are structural rather than advisory. `load_ai4i`
returns `(features, target, failure_modes)` as three separate objects, and the
mode columns exist only in the third. The `AI4IConfig` model carries a validator
that rejects any configuration listing a mode column among the features, so the
separation cannot be undone by editing `configs/data.yaml`. A test asserts the
feature frame contains none of the five, and a second test measures the 99.73%
agreement directly, so the justification for the quarantine is re-derived from the
data on every run rather than taken on trust.

The mode columns remain available, and are genuinely useful — for describing which
failure types dominate, for stratifying error analysis, and for the correlation
heatmap in `notebooks/02_ai4i_eda.ipynb` that makes the leakage visible. They are
simply never model input.

## Class balance and mode distribution

| | Count | Share of rows |
| --- | --- | --- |
| No failure | 9,661 | 96.61% |
| Failure | 339 | 3.39% |

| Failure mode | Occurrences |
| --- | --- |
| HDF — heat dissipation | 115 |
| OSF — overstrain | 98 |
| PWF — power | 95 |
| TWF — tool wear | 46 |
| RNF — random | 19 |

The modes sum to 373 against 339 failures, because some rows trip more than one
mode simultaneously.

## Missingness

There is none: no missing values in any column, in either the features or the
target. This is a consequence of the data being synthetic, and it is worth stating
explicitly in the report — the absence of an imputation step is a property of the
dataset, not an omission in the pipeline.

## Other pitfalls

**Column naming varies by source.** The UCI CSV carries unit suffixes in its
headers (`Air temperature [K]`, `Rotational speed [rpm]`); `ucimlrepo` returns the
same columns without them (`Air temperature`, `Rotational speed`). Code written
against one source breaks silently against the other, usually as a `KeyError`
during feature selection. The loader normalises both to canonical snake_case names
with the unit folded in — `air_temperature_k`, `rotational_speed_rpm` — via a
regex for the bracketed form plus an explicit alias table for the stripped form,
since no rule can infer a unit that was never written down.

**`UID` and `Product ID` are identifiers, not features.** `UID` is a row counter,
so it correlates with nothing; `Product ID` embeds the quality letter and a serial
number, so a model can memorise it. Neither is included in the feature frame.

**Do not read real failure rates off this data.** The 3.39% positive rate is a
property of the generator's chosen thresholds. It is the right thing to train on
and the wrong thing to cite as an industrial base rate.

## Preprocessing applied

1. Fetch via `ucimlrepo` (dataset 601), falling back to the UCI archive zip.
2. Normalise column names to canonical snake_case with units folded in.
3. Cache the full frame to `data/interim/ai4i.parquet`.
4. Split into three objects: the five process measurements plus one-hot product
   type as `features`; `machine_failure` as `target`; the five mode columns as
   `failure_modes`, for analysis only.
5. Assert that no mode column reached the feature frame.

Scaling is deliberately *not* applied at load time. The five measurements are on
very different scales (temperatures near 300 K, speed near 1,500 rpm, wear in
minutes), so standardisation is required — but it must be fitted on the training
fold alone, which makes it the training script's responsibility rather than the
loader's. Doing it here would silently fit on the whole dataset, which is the same
class of mistake as the failure-mode leak, only quieter.
