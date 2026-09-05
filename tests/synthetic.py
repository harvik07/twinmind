"""Synthetic stand-ins for the raw datasets, matching the real schemas exactly.

These let the preprocessing tests run on a clean checkout with no downloads. They
are deliberately small but structurally faithful: the same column names, dtypes
and quirks (weekend gaps in oil, transferred holidays, short C-MAPSS units) that
the real files have, so a test passing here means something.

Anything that must be checked against the *real* data is marked ``@pytest.mark.data``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# C-MAPSS
# --------------------------------------------------------------------------- #
def make_cmapss(
    n_units: int = 6,
    min_cycles: int = 40,
    max_cycles: int = 120,
    n_sensors: int = 21,
    n_constant: int = 3,
    seed: int = 0,
    short_unit_cycles: int | None = None,
) -> pd.DataFrame:
    """A C-MAPSS-shaped training frame.

    The first ``n_constant`` sensors are flat (so constant detection has something
    to find) and the rest drift with cycle. If ``short_unit_cycles`` is given, the
    final unit is truncated to that many cycles to exercise window padding.
    """
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []

    for unit in range(1, n_units + 1):
        n = int(rng.integers(min_cycles, max_cycles + 1))
        if short_unit_cycles is not None and unit == n_units:
            n = short_unit_cycles
        cycle = np.arange(1, n + 1)
        progress = cycle / n

        block = {"unit": np.full(n, unit, dtype="int64"), "cycle": cycle.astype("int64")}
        for s in range(1, 4):
            block[f"op_setting_{s}"] = rng.normal(0, 0.002, n)
        for s in range(1, n_sensors + 1):
            if s <= n_constant:
                block[f"sensor_{s}"] = np.full(n, 10.0 + s)
            else:
                block[f"sensor_{s}"] = (
                    100.0 + 5.0 * s + 20.0 * progress**2 + rng.normal(0, 0.4, n)
                )
        rows.append(pd.DataFrame(block))

    return pd.concat(rows, ignore_index=True)


def make_cmapss_multi_regime(
    n_units: int = 6,
    n_regimes: int = 6,
    cycles: int = 90,
    n_sensors: int = 21,
    seed: int = 0,
) -> pd.DataFrame:
    """A frame whose op settings form ``n_regimes`` well-separated clusters.

    Each regime shifts every sensor by a large regime-specific offset, so a global
    z-score leaves the degradation trend buried and only within-regime
    standardisation recovers it.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[i * 10.0, i * 5.0, i * 2.0] for i in range(n_regimes)])
    rows: list[pd.DataFrame] = []

    for unit in range(1, n_units + 1):
        cycle = np.arange(1, cycles + 1)
        regime = rng.integers(0, n_regimes, cycles)
        progress = cycle / cycles
        block: dict[str, np.ndarray] = {
            "unit": np.full(cycles, unit, dtype="int64"),
            "cycle": cycle.astype("int64"),
            # Ground-truth regime, for tests to group by. Never a model input.
            "true_regime": regime.astype("int64"),
        }
        settings = centres[regime] + rng.normal(0, 0.05, (cycles, 3))
        for s in range(3):
            block[f"op_setting_{s + 1}"] = settings[:, s]
        for s in range(1, n_sensors + 1):
            block[f"sensor_{s}"] = (
                50.0 + 100.0 * regime + 10.0 * progress + rng.normal(0, 0.3, cycles)
            )
        rows.append(pd.DataFrame(block))

    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Favorita
# --------------------------------------------------------------------------- #
def make_favorita(
    n_stores: int = 4,
    families: tuple[str, ...] = ("GROCERY I", "BEVERAGES", "PRODUCE"),
    start: str = "2015-01-01",
    days: int = 420,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """The six Favorita tables, with the real column names and quirks.

    Reproduced faithfully: oil has weekday-only coverage (weekend NaNs), and
    holidays_events includes a transferred Holiday, its matching Transfer row, a
    Bridge and the Work Day that repays it, plus Regional and Local entries.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")

    cities = ["Quito", "Guayaquil", "Cuenca", "Ambato"]
    states = ["Pichincha", "Guayas", "Azuay", "Tungurahua"]
    stores = pd.DataFrame(
        {
            "store_nbr": np.arange(1, n_stores + 1, dtype="int64"),
            "city": [cities[i % len(cities)] for i in range(n_stores)],
            "state": [states[i % len(states)] for i in range(n_stores)],
            "type": [["A", "B", "C", "D"][i % 4] for i in range(n_stores)],
            "cluster": [(i % 5) + 1 for i in range(n_stores)],
        }
    )

    records: list[pd.DataFrame] = []
    for store in range(1, n_stores + 1):
        for fi, family in enumerate(families):
            trend = np.linspace(0, 20, days)
            weekly = 15 * np.sin(2 * np.pi * np.arange(days) / 7)
            yearly = 25 * np.sin(2 * np.pi * np.arange(days) / 365.25)
            base = 100.0 * (store + fi + 1)
            sales = base + trend + weekly + yearly + rng.normal(0, 5, days)
            records.append(
                pd.DataFrame(
                    {
                        "id": np.arange(len(dates)) + store * 100000 + fi * 10000,
                        "date": dates,
                        "store_nbr": np.full(days, store, dtype="int64"),
                        "family": family,
                        "sales": np.clip(sales, 0, None),
                        "onpromotion": rng.integers(0, 10, days).astype("int64"),
                    }
                )
            )
    train = pd.concat(records, ignore_index=True)

    test = train.tail(0).drop(columns=["sales"]).copy()

    # Oil trades on weekdays only; weekends are genuinely absent from the file.
    weekdays = dates[dates.dayofweek < 5]
    oil = pd.DataFrame(
        {
            "date": weekdays,
            "dcoilwtico": 50.0 + np.cumsum(rng.normal(0, 0.3, len(weekdays))),
        }
    )

    holidays = pd.DataFrame(
        [
            # A holiday that was MOVED: it did not happen on this date.
            {"date": dates[10], "type": "Holiday", "locale": "National",
             "locale_name": "Ecuador", "description": "Primer Grito moved", "transferred": True},
            # ...it was actually observed here.
            {"date": dates[12], "type": "Transfer", "locale": "National",
             "locale_name": "Ecuador", "description": "Primer Grito", "transferred": False},
            # An ordinary national holiday.
            {"date": dates[40], "type": "Holiday", "locale": "National",
             "locale_name": "Ecuador", "description": "Navidad", "transferred": False},
            # A bridge day and the Work Day that repays it.
            {"date": dates[41], "type": "Bridge", "locale": "National",
             "locale_name": "Ecuador", "description": "Puente", "transferred": False},
            {"date": dates[45], "type": "Work Day", "locale": "National",
             "locale_name": "Ecuador", "description": "Recupero", "transferred": False},
            # Regional and local, which apply to a subset of stores only.
            {"date": dates[60], "type": "Holiday", "locale": "Regional",
             "locale_name": "Pichincha", "description": "Provincializacion",
             "transferred": False},
            {"date": dates[70], "type": "Holiday", "locale": "Local",
             "locale_name": "Guayaquil", "description": "Fundacion", "transferred": False},
            {"date": dates[80], "type": "Additional", "locale": "National",
             "locale_name": "Ecuador", "description": "Navidad-1", "transferred": False},
        ]
    )

    transactions = pd.DataFrame(
        {
            "date": np.tile(dates, n_stores),
            "store_nbr": np.repeat(np.arange(1, n_stores + 1), len(dates)).astype("int64"),
            "transactions": rng.integers(500, 3000, len(dates) * n_stores).astype("int64"),
        }
    )

    return {
        "train": train,
        "test": test,
        "stores": stores,
        "oil": oil,
        "holidays": holidays,
        "transactions": transactions,
    }
