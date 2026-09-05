# Data card — SKAB (Skoltech Anomaly Benchmark)

> **Optional dataset.** SKAB is not required by any TwinMind model. It is fetched
> only with `--dataset skab` or `--dataset all --include-optional`, and nothing in
> the training pipeline depends on it.

## Source and licence

The Skoltech Anomaly Benchmark, published by the Skolkovo Institute of Science and
Technology at `https://github.com/waico/SKAB`. Released under the GNU General
Public License v3.0 — a copyleft licence, which is why this dataset is kept at
arm's length from the main pipeline: it informs design, it does not ship inside a
model.

TwinMind fetches the CSVs directly over `raw.githubusercontent.com`, enumerating
each data folder through the GitHub contents API rather than cloning the
repository, so no git dependency is introduced.

## What it contains

Sensor telemetry from a physical testbed — a water-circulation rig with a pump,
valves and a control system — deliberately driven into fault conditions while
being recorded. Eight sensor channels: two accelerometer RMS values, current,
pressure, temperature, thermocouple, voltage and volume flow rate RMS.

As downloaded and concatenated:

| | Value |
| --- | --- |
| Rows | 37,401 |
| Columns | 13 (8 sensors + datetime + 2 labels + 2 provenance) |
| Experiment groups | 3 (`valve1`, `valve2`, `other`) |
| Distinct experiment files | 34 CSVs |

| Group | Rows | Anomaly rate |
| --- | --- | --- |
| `valve1` | 18,160 | 34.7% |
| `other` | 14,929 | 35.1% |
| `valve2` | 4,312 | 35.2% |

The loader adds `group` and `experiment` columns during concatenation, since file
stems repeat across groups and would otherwise become indistinguishable.

## Target definition

`anomaly` — a binary per-timestamp label, 1 during a fault. A second label,
`changepoint`, marks the transition instants rather than the fault intervals; it is
preserved but unused.

## Class balance

**34.94% of rows are anomalous** — 13,067 of 37,401, and close to uniform across
the three groups.

This is a strikingly high rate for an anomaly benchmark, and it is the most
important thing to understand about SKAB. It does not mean the underlying process
fails a third of the time. It reflects how the data was collected: each file is a
short experiment deliberately run into a fault, so the recordings are concentrated
around the failure rather than sampled from normal operation. Treating 35% as a
base rate for anomaly detection would be a serious misreading.

## Missingness

None. All 37,401 rows are complete across all columns.

## Known pitfalls

**Semicolon delimiter.** The CSVs use `;`, not `,`. A default `read_csv` yields a
single-column frame of concatenated strings — which does not raise, and is easy to
miss. The loader passes `sep=";"` explicitly.

**The anomaly rate is an artefact of collection.** As above: 35% describes the
recording protocol, not the process.

**Filenames collide across groups.** Each group contains files named `1.csv`,
`2.csv` and so on. Concatenating without provenance columns silently merges
unrelated experiments into what looks like one continuous series.

**Timestamps are per-experiment.** Each file has its own `datetime` range. The
concatenated frame is *not* a single continuous time series, and any windowing must
be done within `(group, experiment)`, exactly as C-MAPSS windowing is done within
`unit`.

## Why it is here

SKAB is not training data for any TwinMind model. It is included as a reference
point for the digital twin's failure process: it is real physical-testbed telemetry
of a pump-and-valve system degrading, which is a closer analogue to the simulated
production line than either the aero-engine data (C-MAPSS) or the synthetic
milling data (AI4I).

Its intended use is qualitative validation — checking that the sensor-degradation
patterns the twin produces are of a plausible shape — rather than fitting anything.
Being GPL-3.0, it also should not be redistributed inside this repository, which is
a second reason to keep the dependency optional and explicit.

## Preprocessing applied

1. Enumerate the CSVs in each configured group through the GitHub contents API;
   stream each to `data/raw/skab/<group>/<name>.csv`.
2. Read with `sep=";"`, normalise column names to snake_case, parse `datetime`.
3. Prepend `group` and `experiment` provenance columns.
4. Concatenate to one long frame; cast `anomaly` to `int64`.
5. Cache to `data/interim/skab.parquet`.
