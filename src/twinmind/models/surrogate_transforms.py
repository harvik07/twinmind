"""Per-KPI target transforms, and the inverse that enforces validity.

Each KPI lives on a different support, and training all eight on a raw linear
head would let the network predict a negative unit cost or an OEE of 1.4. The
optimiser takes an argmax over thousands of these, so an impossible value is
not a small error -- it is a scenario that looks artificially good and gets
recommended.

Three families, chosen by what the KPI actually is:

* **Bounded in [0, 1]** -- ``oee``, ``stockout_probability``,
  ``on_time_delivery_rate``. The mean head ends in a sigmoid, so the bound holds
  by construction rather than by clipping afterwards. Trained in logit space so
  the loss is not squashed near the ends, where a lot of these values sit:
  on-time delivery clusters near 0.8 and stockout near 0.2.
* **Non-negative and heavy-tailed** -- ``units_produced``, ``unit_cost``,
  ``downtime_hours``, ``unplanned_failures``. Trained in ``log1p`` space. These
  span orders of magnitude across the scenario box (unit cost runs from about 20
  to several hundred when a line barely produces), and a linear-space loss would
  spend all its capacity on the large values. ``log1p`` rather than ``log``
  because ``unplanned_failures`` is legitimately 0 in most scenarios.
* **Signed** -- ``profit``. Standardised, not transformed. It is genuinely
  negative for a large part of the box (the landscape search found profits from
  -£620k to +£493k), so any non-negative transform would be wrong, and the
  distribution is near enough symmetric that standardising is sufficient.

THE STD TRANSFORMS WITH THE MEAN, AND THAT IS THE SUBTLE PART.

The network predicts a mean and a std **in transformed space**. Mapping the mean
back is just the inverse function, but the std has to be mapped through the
derivative -- a std of 0.1 in log space is not a std of 0.1 in units. The
delta-method factor is applied in :func:`inverse_std`. Getting this wrong makes
the calibration numbers meaningless while leaving the point predictions fine,
which is exactly the kind of error that survives review.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from twinmind.models.surrogate import KPI_ORDER

TransformKind = Literal["log1p", "logit", "standard"]

#: How each KPI is trained. Frozen with the weights: changing one silently
#: invalidates a saved ensemble, so ``config.json`` records it.
TRANSFORMS: dict[str, TransformKind] = {
    "units_produced": "log1p",
    "oee": "logit",
    "unit_cost": "log1p",
    "profit": "standard",
    "stockout_probability": "logit",
    "downtime_hours": "log1p",
    "unplanned_failures": "log1p",
    "on_time_delivery_rate": "logit",
}

#: Keeps the logit finite at exactly 0 or 1, both of which occur: a scenario
#: with no demand has stockout_probability 0 and on_time_delivery_rate 1.
EPSILON = 1e-4


def forward(values: NDArray[np.float64], kind: TransformKind) -> NDArray[np.float64]:
    """Map a KPI's raw values into training space."""
    if kind == "log1p":
        return np.log1p(np.clip(values, 0.0, None))
    if kind == "logit":
        clipped = np.clip(values, EPSILON, 1.0 - EPSILON)
        return np.log(clipped / (1.0 - clipped))
    return np.asarray(values, dtype="float64")


def inverse(values: NDArray[np.float64], kind: TransformKind) -> NDArray[np.float64]:
    """Map a prediction back to the KPI's own units."""
    if kind == "log1p":
        return np.expm1(np.clip(values, -50.0, 50.0))
    if kind == "logit":
        return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))
    return np.asarray(values, dtype="float64")


def inverse_std(
    std: NDArray[np.float64], mean_in_space: NDArray[np.float64], kind: TransformKind
) -> NDArray[np.float64]:
    """Map a std from training space back to the KPI's units, via the derivative.

    The delta method: for a monotone ``g``, ``sd(g(X)) ≈ |g'(mu)| sd(X)``. For
    ``log1p`` the inverse is ``expm1`` with derivative ``exp(mu)``; for the
    logit the inverse is the sigmoid with derivative ``p(1 - p)``.

    Treating the transformed std as if it were already in units is the easy
    mistake and a quiet one: the point predictions stay correct while every
    calibration and coverage number becomes meaningless.
    """
    std = np.clip(np.asarray(std, dtype="float64"), 0.0, None)
    if kind == "log1p":
        return std * np.exp(np.clip(mean_in_space, -50.0, 50.0))
    if kind == "logit":
        probability = inverse(mean_in_space, "logit")
        return std * probability * (1.0 - probability)
    return std


def forward_targets(targets: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform an ``[N, 8]`` label matrix column by column."""
    out = np.empty_like(np.asarray(targets, dtype="float64"))
    for index, kpi in enumerate(KPI_ORDER):
        out[:, index] = forward(np.asarray(targets)[:, index], TRANSFORMS[kpi])
    return out


def inverse_predictions(
    mean: NDArray[np.float64], std: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Back-transform an ``[N, 8]`` mean and std, then enforce KPI validity."""
    from twinmind.models.surrogate import enforce_validity

    mean = np.asarray(mean, dtype="float64")
    std = np.asarray(std, dtype="float64")
    out_mean = np.empty_like(mean)
    out_std = np.empty_like(std)
    for index, kpi in enumerate(KPI_ORDER):
        kind = TRANSFORMS[kpi]
        out_mean[:, index] = inverse(mean[:, index], kind)
        out_std[:, index] = inverse_std(std[:, index], mean[:, index], kind)
    return enforce_validity(out_mean), np.clip(out_std, 0.0, None)


def target_statistics(
    targets_in_space: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-KPI mean and std of the transformed labels, from TRAINING rows only.

    Applied on top of the transform so every head sees a roughly unit-scale
    target. Without it, ``profit`` (hundreds of thousands) would dominate the
    summed loss and the bounded KPIs would barely train.
    """
    mean = targets_in_space.mean(axis=0)
    std = targets_in_space.std(axis=0, ddof=1)
    return mean, np.where(std < 1e-12, 1.0, std)
