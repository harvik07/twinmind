# Data card — Corporación Favorita Store Sales

> **Status: not yet downloaded.** This dataset requires Kaggle credentials and
> acceptance of the competition rules, neither of which was available when the
> data layer was built. Everything below describes the published schema and the
> pipeline that has been written and tested against it; the shape and volume
> figures will be filled in on first successful download. The feature pipeline is
> exercised by a synthetic fixture with the identical schema
> (`tests/synthetic.py`), so its logic is tested — but no claim here about
> *observed* Favorita values has been verified against the real file.

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

The published dataset covers 54 stores × 33 product families = 1,782 series, daily
from 2013-01-01 to 2017-08-15, for roughly 3 million rows.

TwinMind reduces this to the **top 200 series by total volume** (`favorita.top_series`).
Most of the 1,782 are near-zero sellers whose forecast error has no bearing on the
digital twin's order book, and training on all of them costs a Kaggle GPU session
for no benefit.

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

## To complete this card

Run `python scripts/download_data.py --dataset favorita` with valid credentials and
accepted rules, then re-run `notebooks/03_favorita_eda.ipynb`. The notebook detects
the real data automatically and prints the observed row count, series count, date
range and per-fold sizes; those figures replace this section.
