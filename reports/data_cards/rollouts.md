# Data card — surrogate rollouts

The supervised dataset Model C trains on: scenarios evaluated by the digital
twin, with their KPI means and replication noise.

**What these rows describe is the twin, not a factory.** Every row inherits
every assumption in `configs/twin.yaml`, which is why the manifest stores all 15
of them verbatim. A surrogate trained on these rows emulates a simulator whose
`demand_capacity_ratio` is worth 45% of profit and whose warning threshold was
chosen to keep the landscape non-degenerate. See
[twin_design.md](reports/model_cards/twin_design.md) §0.

---

## 1. Design

### A crossed block, not a flat sample

| | |
|---|---|
| Contexts (blocks) | 512 |
| Scenarios per context | 64 |
| **Training scenarios** | **32,768** |
| Replications per scenario | 10, independent seeds |
| **Training twin runs** | **327,680** |

The agent's job is to rank scenarios **within one plant state**, so the data
must contain many scenarios per context. One scenario per context would teach
the surrogate the marginal effect of a scenario averaged over states, and almost
nothing about which plan is better *here*.

### One Sobol sequence, partitioned into blocks

Scenarios come from a single scrambled Sobol sequence over the 18 dimensions in
`SCENARIO_DIMS` order, scaled to the `twin.yaml` bounds. Context block *k*
consumes points `[64k, 64k + 64)`.

Two things follow, and both are load-bearing:

- Sobol is balanced in prefixes of 2^m, so **any prefix of complete blocks is
  itself a valid space-filling design**. That makes a multi-hour job stoppable
  at any point, and makes the 2^11–2^15 learning curve trainable from nested
  prefixes rather than five separate samples.
- A scenario's **global Sobol index is a stable id**, so train/test disjointness
  is asserted on integers rather than on floating-point comparisons.

`shift_count` and `supplier_choice` are rounded to integers after scaling.

### Replication: independent seeds, deliberately not CRN

Each scenario runs 10 replications with seeds from
`SeedSequence((seed_root, scenario_index))`.

Common random numbers are right for *comparing* scenarios and wrong for
*training data*. Under CRN every scenario shares the same luck, so that shared
noise becomes a reproducible function of the inputs and the surrogate learns it
as signal. The stored label is the mean over reps; the stored std is the
per-replication sample std, which the noise ceiling needs.

`SeedSequence((root, index))` hashes the pair, so neighbouring scenarios get
unrelated streams — unlike `root + index`, which produces correlated starting
states.

### Chronological split on context origins

| Split | Origin window |
|---|---|
| Training contexts | 2016-01-01 … 2017-06-20 |
| Test contexts | 2017-06-21 … 2017-08-15 |

Contexts differ in machine wear, stock and the demand forecast, so a random
split would let the surrogate see the same forecast origin in both halves. A
test asserts `max(train origins) < min(test origins)`.

---

## 2. Test splits

Different Sobol scramble, later origins, 30 reps, seeds disjoint from training.

| Split | Shape | Reps | Purpose |
|---|---|---|---|
| **T1** | 40 contexts × 25 scenarios = 1,000 | 30 | i.i.d. accuracy, and generalisation to unseen contexts |
| **T2** | 1 context × 1,000 scenarios, **twice** | 30 each | ranking within one state, with a twin-vs-twin ceiling |
| **OOD** | 300 scenarios, ≥3 dimensions at a bound | 30 | extrapolation |

**T2 is evaluated twice with independent seeds.** The agreement between the two
30-rep estimates is the **ceiling**: the ranking agreement a *perfect* surrogate
would show, because the test truth is itself noisy. Reporting a Spearman ρ
without that ceiling would make an excellent surrogate look mediocre.

**The T2 context is chosen, not arbitrary.** 64 candidate contexts are sampled
at the first test origin and the one whose state vector is nearest the
element-wise median (normalised by per-feature spread) wins. A context with
every machine nearly dead would make the maintenance dimensions dominate and
flatter the surrogate on an unusual problem.

**OOD is built to be genuinely outside the training density.** Sobol points with
3–6 randomly chosen coordinates snapped to 0 or 1 before scaling. An "unlikely"
interior point still sits inside the convex hull of the training data and says
nothing about extrapolation.

### Leak guards

- Every split's manifest records what it is, and `load_rollouts` **refuses** to
  return a non-training split unless the caller passes `allow_test=True`.
  Training code never passes it.
- The validation split holds out **whole context blocks**, not random rows.
  Splitting rows would leave 63 of a block's 64 scenarios in training and one in
  validation — measuring interpolation at an already-seen state.
- State standardisation statistics are an explicit argument to `encode`, computed
  from training rows only. Computing them over the whole file is the convenient
  thing to do and is a leak.
- Tests assert scenario ids, context blocks **and realised seed integers** are
  disjoint between train and test.

---

## 3. Storage

One Parquet shard per context block in `data/processed/rollouts/` (gitignored —
derived data that regenerates from the manifest).

Each row carries: the 18 scenario values, the 15 context state features, the 8
KPI means, the 8 KPI stds, `n_reps`, the replication seeds as JSON, the context
block and the global scenario index.

**KPI columns are prefixed `kpi_`.** Three state fields are named
`demand_q10_mean`, `demand_q50_mean` and `demand_q90_mean`, so a loader
selecting targets by `endswith("_mean")` would silently pull three context
features in as labels. The prefix makes the target set unambiguous by
construction rather than by care.

**Writes are atomic** — temp file then `os.replace`. An interrupted multi-hour
job would otherwise leave a truncated parquet that the resume logic counts as
done and silently trains on. `valid_shard` checks the row count, not just
existence.

**Resume is manifest-checked.** The manifest records the twin config hash, the
health checkpoint and cache hashes, the demand config hash, the git sha, the
demand scale and **all 15 declared assumption values**. Resuming against a
changed twin is refused, because rollouts generated under
`demand_capacity_ratio = 0.85` and 0.64 describe different simulators and
concatenating them would train one model on two. `git_sha` and `created`
deliberately do not invalidate: re-running after an unrelated commit is fine.

---

## 4. Runtime, measured on this machine

6 physical cores (12 logical), 7.3 GB RAM, Windows, `spawn` start method.

| Configuration | Throughput |
|---|---|
| Single core | 7.46 runs/s (134 ms/run) |
| 6 workers, no grid cache | 31.9 runs/s |
| 6 workers + grid cache | 27.5 runs/s |
| **10 workers + grid cache** | **38.8 runs/s** |

**The pool was memory-bound before it was CPU-bound, and that was measured, not
guessed.** Each worker held **1.33 GB**, of which **0.71 GB was the featurised
Favorita frame** (336,800 × 47) and 0.15 GB the LightGBM boosters. On 7.3 GB
that caps the pool at about four workers; an 11-worker pool **died outright,
producing zero shards and exiting 0**.

The twin never touches that frame at run time — it asks the sampler for one
calibrated quantile grid per `(series, origin)`, and the grid depends on neither
the scenario nor the seed. `scripts/build_demand_cache.py` precomputes all 556
origins into **805 KB**, and a worker then holds **0.15 GB instead of 1.33 GB**,
a 9× reduction. A test asserts the cached and full models give **bit-identical**
twin output.

### The cache is not optional, and the code now enforces that

The first test-splits run died with a numpy `_ArrayMemoryError` in a worker,
despite `--demand-cache` being passed and the cache file being present and
readable. The cause was not the cache: `demand_cache` was an **optional keyword**
on `generate()`, the training call site passed it and **all three test-split
call sites did not**, so every test-splits worker rebuilt the full frame.

Four changes make that failure unexpressible rather than merely fixed:

- Shared settings (workers, replications, cache) are one **required** argument,
  so a call site cannot pass some and forget others.
- **One task-building function** serves both the training and test-splits paths.
- A cache that is asked for and cannot be read **raises**, naming the file.
  There is no fallback to the full frame.
- More than one worker with no cache is **refused** unless `--allow-full-frame`
  is passed, and the default cache path is **auto-detected** when present.

Each worker now reports its resident memory with every shard, so the footprint
is visible in the first shard rather than inferred from a crash in the fiftieth.
Verified on a 10-worker test-splits smoke run: **0.15 GB mean and peak per
worker, ~1.5 GB across the pool.**

### Projected wall-clock

| Job | Runs | Projected |
|---|---:|---:|
| Training, 32,768 × 10 | 327,680 | **2.35 h** |
| T1, 1,000 × 30 | 30,000 | 0.21 h |
| T2, 2 × 1,000 × 30 | 60,000 | 0.43 h |
| OOD, 300 × 30 | 9,000 | 0.06 h |
| **Total** | **426,680** | **≈ 3.05 h** |

Effective parallelism is 5.4× on 10 workers against 6 physical cores, which is
about what hyperthreading delivers. An earlier version of the projection used an
assumed 0.85 efficiency and was wrong in two directions at once: it ignored the
physical/logical core split and it ignored that the job was memory-bound.

---

## 5. Reproduce

```bash
python scripts/build_demand_cache.py
python scripts/generate_rollouts.py --measure
python scripts/generate_rollouts.py --smoke
python scripts/generate_rollouts.py --contexts 512 --scenarios-per-context 64 \
    --n-reps 10 --workers 10 --demand-cache data/processed/rollouts/demand_grids.npz
python scripts/generate_rollouts.py --test-splits --workers 10 \
    --demand-cache data/processed/rollouts/demand_grids.npz
```

Both generation commands are resumable: rerun the same command after an
interruption and completed shards are skipped.

---

## 6. Limitations

- **One demand series.** The twin makes one product, so every context shares
  `1_GROCERY I` and differs only in origin. Context diversity comes from machine
  wear, stock and the forecast date, not from product mix.
- **512 origins over 18 months**, spread deterministically rather than sampled,
  so origin coverage is even but not random.
- **The surrogate inherits the twin's assumptions,** including the ones the
  twin's own sensitivity study flags as dominant. A surrogate that matches the
  twin perfectly is still only as good as the twin.
- **10 reps per training scenario** means each label carries a non-trivial
  standard error — which is the point of storing the std, and is why the
  evaluation is noise-aware rather than comparing against R² = 1.
