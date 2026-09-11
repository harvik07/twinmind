# Data card — Corporación Favorita Store Sales

> **Status: downloaded and verified** (2026-09-10). Every figure below is measured
> from the real files, not predicted. Acquisition needed a client upgrade: Kaggle
> now issues `KGAT_` access tokens, which `kaggle` 1.x cannot use — see
> "Authentication" below.

## Source and licence

The "Store Sales — Time Series Forecasting" Kaggle competition, using point-of-sale
data from Corporación Favorita, a large Ecuadorian grocery retailer. TwinMind
fetches it through the Kaggle API from competition
`store-sales-time-series-forecasting`.

Competition data is governed by the competition rules rather than an open licence,
and access requires both an API token and explicit acceptance of those rules on the
website. This is worth planning around: an API token alone is not sufficient, and
the failure mode is an HTTP 403 that says nothing about the cause. The downloader
therefore catches 403 specifically and prints the sign-up URL and the exact steps,
because the raw exception is actively misleading — it looks like an authentication
problem when it is a rules-acceptance problem.

## What it contains

Six tables. `train.csv` is the core: daily unit sales per store per product family,
along with a count of items on promotion. `stores.csv` gives store city, state, type
and cluster; `oil.csv` gives the daily WTI price (Ecuador's economy is
oil-dependent); `holidays_events.csv` lists national, regional and local holidays;
`transactions.csv` gives per-store daily transaction counts; `test.csv` is the
competition's unlabelled horizon.

**Measured on the downloaded files:**

| Table | Rows | Columns |
| --- | --- | --- |
| `train.csv` | 3,000,888 | 6 |
| `test.csv` | 28,512 | 5 |
| `transactions.csv` | 83,488 | 3 |
| `oil.csv` | 1,218 | 2 |
| `holidays_events.csv` | 350 | 6 |
| `stores.csv` | 54 | 5 |

54 stores × 33 product families = **1,782 series**, daily from **2013-01-01 to
2017-08-15**, 124.8 MB across 7 files.

TwinMind reduces this to the **top 200 series by total volume** (`favorita.top_series`).
Those 200 series — 11.2% of the 1,782 — carry **75.5% of all units sold**, which is
the ratio that justifies the cut: the long tail is near-zero sellers whose forecast
error has no bearing on the digital twin's order book, and training on all of them
costs a Kaggle GPU session for no benefit.

After feature building the frame is **336,800 rows × 47 columns** (200 series ×
1,684 days), assembled in about 4 seconds. Of those rows 8.21% fall on a holiday
and 1,000 on a Work Day.

## Target definition

`sales` — units sold for a given store, product family and date. It is continuous
and non-negative, and includes fractional values, since some families are sold by
weight rather than by count.

Model B forecasts it at three quantiles (q10, q50, q90) over a 28-day horizon, so
the digital twin's demand process can sample from a distribution rather than
following a point forecast. That choice is what lets stockout probability be a
meaningful KPI rather than an artefact of a single trajectory.

## Class balance

Not applicable — this is a continuous regression target. The relevant distributional
facts are that sales are strongly right-skewed and contain a large number of exact
zeros (a family a store does not stock reads as zero every day, not as missing), and
that both weekly and yearly seasonality are pronounced.

## Missingness

Two structural gaps, both handled explicitly:

**Oil prices are weekday-only.** `oil.csv` has entries for trading days, so weekends
and market holidays are absent from the file entirely — around 30% of calendar days.
`prepare_oil` reindexes onto a complete daily calendar and forward-fills, which is
the correct semantics: on a Sunday the prevailing price *is* Friday's close.
Back-filling would be anticipative, so the single leading gap before the first
observation is filled with the first observed price instead, and a test asserts a
Sunday carries the preceding Friday's value rather than the following Monday's.

**Missing series-days.** Some store-family combinations are absent on some dates,
particularly around the 2016 earthquake and for families a store had not yet begun
stocking. Lag and rolling features are computed per series after sorting by date, so
a gap shortens the available history rather than shifting a lag onto the wrong date.

## Known pitfalls

**Transferred holidays — the main trap in this dataset.** `holidays_events.csv`
encodes five distinct event types, and treating them uniformly misaligns some of
the largest demand spikes in the data:

- A row with `type = "Holiday"` and `transferred = True` **did not happen on the
  date it is listed against**. The holiday was moved; that date was an ordinary
  working day. Flagging it as a holiday is the classic error.
- A row with `type = "Transfer"` carries the date the holiday was *actually*
  observed. These are real holidays.
- `Bridge` days are extra days off joining a holiday to a weekend.
- `Work Day` rows are the **opposite** of a holiday: ordinary working days, usually
  Saturdays, that repay a Bridge. Treating them as holidays inverts the signal
  exactly where it matters most.
- `Additional` rows are extra official holiday days and count as holidays.

`build_holiday_flags` implements all five cases, and four separate tests pin the
behaviour: a transferred holiday is not flagged on its listed date, its Transfer row
is flagged, a Work Day is flagged as a work day and *not* as a holiday, and Bridge
and Additional days are flagged as holidays.

**Holiday locale.** Holidays carry a `locale` of National, Regional or Local, with
`locale_name` naming a state or a city. A local holiday in Guayaquil does not close
stores in Quito. Flags are resolved per store by joining `locale_name` against the
store's city or state, rather than applied nationally; a test asserts a local
holiday reaches only stores in the matching city.

**Leakage through lags and rolling statistics.** This is the failure that looks like
success. Writing `.rolling(7).mean()` on the target includes the current row's `y`
in its own feature, so validation error collapses and live forecasting falls apart.
Every rolling statistic here is computed on the **already-shifted** series, so
`roll_mean_7` at date *t* averages *t−7 … t−1* strictly. Lags are `shift(k)` with
`k ≥ 1`, and a lag of 0 raises rather than being silently accepted.

The general guard is `test_features_are_non_anticipative_under_truncation`: build
features on the full frame, remove the last 60 days of raw input, rebuild, and assert
every overlapping row is bit-identical. Any feature that peeked into the future would
change when that future was removed.

**Random splits.** A K-fold on a time series trains on the future to predict the
past. `make_rolling_origin_splits` walks forward: fold *i* validates on a 28-day
window and trains only on data strictly before it, with a test asserting
`max(train.ds) < min(val.ds)` for every fold.

**Earthquake regime shift.** The April 2016 Ecuador earthquake produced a sharp,
sustained demand anomaly. It is genuine and should not be cleaned away, but any fold
straddling it will show inflated error, which is worth noting when reporting
per-fold metrics.

## Preprocessing applied

1. Download the competition archive; extract the six CSVs.
2. Normalise column names to snake_case; parse date columns to `datetime64[ns]`.
   Cache each table to `data/interim/favorita_<table>.parquet`.
3. Reshape `train` to Nixtla long format: `unique_id` = `"<store_nbr>_<family>"`,
   `ds` = date, `y` = sales.
4. Select the top *N* = 200 series by total volume.
5. Join store metadata (city, state, type, cluster) as categoricals.
6. Add calendar features: day of week, day of month, month, ISO week, year, weekend
   flag, and a payday flag for the 15th and month-end (Ecuadorian public-sector
   wages are paid on those days and sales visibly respond).
7. Add Fourier terms — weekly K = 3 and yearly K = 5, phased from a fixed epoch so
   they are identical across series and folds. Fourier pairs represent smooth
   seasonality in a handful of columns and extrapolate to unseen dates, which
   one-hot day-of-year cannot.
8. Resolve holiday flags per store with the full transferred/bridge/work-day logic.
9. Forward-fill oil onto the daily calendar.
10. Add lags {1, 7, 14, 28} and shifted rolling mean/std over {7, 28}, per series.
11. Sort by `(unique_id, ds)`.

The output is a long frame carrying `unique_id`, `ds`, `y` and roughly 40 exogenous
columns, ready to hand to `NeuralForecast` without further reshaping.

## Authentication

Kaggle runs two credential schemes and they are not interchangeable:

* **Access token** — a single `KGAT_`-prefixed value in `KAGGLE_API_TOKEN` or
  `~/.kaggle/access_token`. Supported from `kaggle` 2.x.
* **Legacy API key** — `KAGGLE_USERNAME` plus a 32-character hex `KAGGLE_KEY`, sent
  as HTTP Basic auth. This is what `kaggle.json` holds.

Putting a `KGAT_` token in `KAGGLE_KEY` on a 1.x client sends it as a Basic-auth
password and the server answers **401** — which reads as a bad password rather than
the wrong scheme, and is how this dataset stayed undownloaded for a week.
`twinmind.data.kaggle_auth` resolves the scheme, recognises a token that has been
put in the key variable, and separates 401 (credentials rejected) from 403 (rules
not accepted). Verify with:

```bash
python scripts/download_data.py --check-auth
```

Observed rolling-origin folds (3 folds, horizon 28):

| Fold | Train rows | Train ends | Val rows | Val window |
| --- | --- | --- | --- | --- |
| 0 | 320,000 | 2017-05-23 | 5,600 | 2017-05-24 → 2017-06-20 |
| 1 | 325,600 | 2017-06-20 | 5,600 | 2017-06-21 → 2017-07-18 |
| 2 | 331,200 | 2017-07-18 | 5,600 | 2017-07-19 → 2017-08-15 |
