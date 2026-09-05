# Data card — NASA C-MAPSS Turbofan Engine Degradation

## Source and licence

The Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan
degradation dataset, published by the NASA Ames Prognostics Center of Excellence
and distributed through the NASA Prognostics Data Repository. TwinMind fetches it
from the repository archive at
`https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`.

The data is a US Government work released into the public domain, with the usual
request that the originating paper be cited: A. Saxena, K. Goebel, D. Simon and
N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure
simulation", *International Conference on Prognostics and Health Management*, 2008.
That paper and the dataset readme are preserved in `data/raw/cmapss/docs/`.

A practical note on acquisition: the published archive contains a second zip
nested inside the first, so a single extraction pass leaves the actual `.txt`
files still packed. The downloader unpacks recursively and then flattens the
result, because the inner archive's directory name contains spaces and a leading
digit that make it awkward to reference from code.

## What it contains

Each of the four subsets is an independent population of simulated turbofan
engines run from a healthy state to failure. A row is one operating cycle of one
engine, carrying three operating-condition settings and twenty-one sensor
measurements. Engines differ in their initial wear and manufacturing variation,
so trajectories vary substantially in length even within a subset.

| Subset | Train rows | Train units | Test rows | Test units | Shortest run | Median run | Longest run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FD001 | 20,631 | 100 | 13,096 | 100 | 128 | 199 | 362 |
| FD002 | 53,759 | 260 | 33,991 | 259 | 128 | 199 | 378 |
| FD003 | 24,720 | 100 | 16,596 | 100 | 145 | 220 | 525 |
| FD004 | 61,249 | 249 | 41,214 | 248 | 128 | 234 | 543 |

The four subsets differ deliberately along two axes. FD001 and FD003 are flown at
a single operating condition; FD002 and FD004 span six. FD001 and FD002 have one
fault mode (high-pressure compressor degradation); FD003 and FD004 add a second
(fan degradation). FD004 is therefore the hardest and FD001 the easiest, and
reporting only FD001 results — as a good deal of published work does — overstates
how well a method generalises.

Note that FD002 and FD004 have one fewer unit in test than in train (259 vs 260,
248 vs 249). This is a property of the published files rather than a loading
error; the loader asserts that each `RUL_*.txt` has exactly as many rows as its
test file has units, and that assertion passes.

## Target definition

The prediction target is remaining useful life: the number of operating cycles
between the current cycle and failure. For a training unit this is directly
observable, because the trajectory runs to failure.

TwinMind uses the standard **piecewise-linear** target, capped at
`R_early = 125` cycles (`cmapss.rul_cap` in `configs/data.yaml`):

```
RUL = min(max_cycle_of_unit - cycle, 125)
```

The reasoning matters enough to state plainly. A raw linear target asserts that an
engine at cycle 5 of a 300-cycle life is "295 cycles from failure". Nothing in the
sensor readings supports that claim: early in life a healthy engine looks
essentially identical whether it has 200 or 300 cycles left, because degradation
has not begun. Regressing on that difference forces the network to fit noise, and
because the early-life region contains most of the rows, the resulting error
dominates the loss. Holding the target flat at 125 until the engine enters its
degradation phase asks the model only for what the data can actually tell it. The
convention traces to Heimes (2008) and is near-universal in the C-MAPSS
literature, so comparisons against published RMSE are only meaningful with it
applied.

The cap binds on a substantial fraction of rows: 39.4% on FD001, 39.5% on FD002,
49.4% on FD003 and 49.2% on FD004.

## Class balance

Not applicable — this is a regression target. The distribution of the capped RUL
is heavily concentrated at the cap itself (the percentages above), with the
remainder spread roughly uniformly over 0–124 cycles. The auxiliary
failure-probability head of Model A is trained on AI4I 2020 instead; see that
data card.

## Missingness

There is none. Every cell of every file is populated. The one artefact worth
knowing about is structural rather than missing data: each line ends with a
trailing separator, so a naïve whitespace parse produces 28 columns of which the
final two are entirely empty. The loader detects this, asserts the surplus
columns are genuinely all-NaN before discarding them, and raises if they are not
— so a future change in the file format fails loudly instead of silently shifting
every column by two.

## Known pitfalls

**Test-set truncation.** This is the one that catches people. Test trajectories
stop at a random point *before* failure, and the ground-truth RUL at that final
cycle is supplied separately in `RUL_<subset>.txt`. Computing RUL on a test frame
the same way as on a training frame — `max_cycle - cycle` — silently assumes every
test engine failed on its last observed cycle, which is false for all of them. The
observed truncation is substantial: mean remaining life at the final test cycle is
75.5 cycles on FD001, 81.2 on FD002, 75.3 on FD003 and 86.6 on FD004, ranging up
to 195. TwinMind provides `compute_test_rul`, which adds the ground-truth offset,
and the loader attaches `rul_at_last_cycle` and `is_last_cycle` to every test
frame so the correct computation is the convenient one.

**Constant sensors differ by subset.** Several sensors never move, and feeding
them to a network wastes capacity and destabilises normalisation. Many
implementations hard-code the FD001 indices. That is wrong for FD002 and FD004,
where the six operating regimes make most of those same sensors informative.
TwinMind detects them from the training split by variance:

| Subset | Sensors dropped |
| --- | --- |
| FD001 | 1, 5, 6, 10, 16, 18, 19 (7 sensors) |
| FD002 | 16 (1 sensor) |
| FD003 | 1, 5, 10, 16, 18, 19 (6 sensors) |
| FD004 | 16 (1 sensor) |

The threshold is `1e-4`, chosen because there is a wide empirical gap either side
of it on every subset: the largest dropped variance is 2.2e-05 and the smallest
kept variance is 3.3e-04. It has to be set above the obvious value of `1e-6`
because FD001's `sensor_6` takes exactly two distinct values, 21.60 and 21.61,
giving it a variance of 1.9e-06 — numerically non-zero but carrying no usable
information, and dropped by every published treatment of this dataset.

**Operating-regime confounding.** On FD002 and FD004 a sensor's raw level is
dominated by which of the six regimes the engine is currently in, not by how
degraded it is. Global z-scoring leaves the degradation trend buried under regime
switching. The fix is to cluster the three operating-setting columns (k-means,
k = 6, fixed seed) and standardise each sensor *within* its cluster.

**Fitting the normaliser on the wrong data.** Clustering or standardising over
train and test together leaks test statistics into training. `OperatingConditionNormaliser`
separates `fit` from `transform`, has no `fit_transform`, refuses to transform
before it has been fitted, and serialises so the exact training statistics can be
reapplied later. A test asserts that transforming a frame shifted by +500 does not
produce a zero-mean output — which it would if `transform` were recomputing
statistics from its argument.

**Splitting within a trajectory.** Cycle 100 and cycle 101 of the same engine are
nearly identical rows. A random row-level split puts one in train and the other in
validation, so the validation score measures interpolation between adjacent cycles
rather than generalisation to an unseen engine. `split_by_unit` splits on engine
id and asserts the two unit sets are disjoint.

## Preprocessing applied

1. Parse the headerless, space-delimited files into 26 columns: `unit`, `cycle`,
   `op_setting_1..3`, `sensor_1..21`. Drop the two phantom trailing columns after
   asserting they are empty. Cast ids to `int64`, measurements to `float64`, sort
   by `(unit, cycle)`.
2. For test splits, join `RUL_<subset>.txt` on unit index and mark each unit's
   final row with `is_last_cycle`.
3. Cache the tidy frame to `data/interim/cmapss_<subset>_<split>.parquet`.
4. Compute the piecewise-linear RUL target, capped at 125.
5. Split the training data by unit into train and validation (`val_fraction`
   0.2, seed 42).
6. Detect and drop near-constant sensors **using the training split only**.
7. Fit the operating-condition normaliser on the training split; apply it to
   validation and test. Six clusters for FD002/FD004, global z-scoring otherwise.
8. Build sliding windows of 30 cycles at stride 1, labelled with the target at the
   window's final cycle.

On step 8, units shorter than the window are left-padded by repeating their first
observation rather than being dropped. This is not a hypothetical case: six FD002
test units and eleven FD004 test units are shorter than 30 cycles, and since the
benchmark protocol scores exactly one prediction per test engine, dropping them
would put 17 engines beyond evaluation. Padding with the first observation rather
than with zeros keeps the padded region on the sensor manifold, so it reads as
"the engine sat at its initial condition" instead of as an impossible reading.
