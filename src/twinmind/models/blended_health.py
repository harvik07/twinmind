"""BlendedHealthModel: the average of the forest and A1, behind the same Protocol.

This is the production hazard model. Neither component wins on its own -- the
forest reaches 12.72 +/- 0.19 on FD001 and the tuned A1 variant 12.76 +/- 0.16,
statistically indistinguishable -- but their mean reaches **12.14 +/- 0.13**, and
beats both on every individual seed rather than on average:

    seed    A1(HCt)   forest    blend
    42      12.627    12.539    12.033
    43      12.703    12.691    12.097
    44      12.941    12.919    12.277

The NASA score improves too, 266 and 245 down to 228, which matters more than the
RMSE because it is asymmetric: it punishes late predictions, and late is the
direction that gets an engine run past its failure point.

Why it works is the ordinary reason ensembles work, and it is worth stating
because it also says when it would stop working. The two models are wrong in
uncorrelated ways: the forest sees six hand-cut summary statistics per sensor and
no temporal ordering at all, while A1 sees the raw ordered window through a
convolutional and recurrent trunk. They fail on different engines, so averaging
cancels part of each one's error. Two models of the *same* family, however well
tuned, would not give this.

The blend is a mixture, so its variance follows the law of total variance --
the mean of the component variances plus the variance of the component means.
The second term is the useful one: when the forest and A1 disagree about an
engine, the blend widens its own interval, which is exactly the behaviour the
twin wants from a hazard it is about to sample from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from twinmind.models.forest_health import ForestHealthModel
from twinmind.models.health import RULPrediction
from twinmind.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BlendConfig:
    """Mixture weights and the shared horizon."""

    #: Weight on the forest. 0.5 is the measured optimum and also the default a
    #: reader would assume; it was not tuned per-seed, which would overfit three
    #: numbers.
    forest_weight: float = 0.5
    horizon_h: int = 30
    rul_cap: int = 125

    def __post_init__(self) -> None:
        if not 0.0 <= self.forest_weight <= 1.0:
            raise ValueError(f"forest_weight must be in [0, 1], got {self.forest_weight}")

    @property
    def network_weight(self) -> float:
        """Weight on the neural component."""
        return 1.0 - self.forest_weight


class BlendedHealthModel:
    """Weighted mean of a forest and a neural health model. Satisfies ``HealthModel``.

    ``network`` is anything exposing ``predict_arrays(windows) -> (mean, std,
    p_fail)``, which both :class:`ForestHealthModel` and the A1 adapter provide.
    """

    def __init__(
        self,
        forest: ForestHealthModel,
        network: object,
        config: BlendConfig | None = None,
    ) -> None:
        if not hasattr(network, "predict_arrays"):
            raise TypeError("network must expose predict_arrays(windows)")
        self.forest = forest
        self.network = network
        self.config = config or BlendConfig()

    def predict_arrays(
        self, windows: NDArray[np.floating]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(rul_mean, rul_std, p_fail)`` for ``[N, T, C]`` windows."""
        array = np.asarray(windows)
        if array.ndim != 3:
            raise ValueError(f"expected a batch [N, T, C], got {array.shape}")

        w = self.config.forest_weight
        forest_mean, forest_std, forest_p = self.forest.predict_arrays(array)
        net_mean, net_std, net_p = self.network.predict_arrays(array)

        mean = np.clip(w * forest_mean + (1 - w) * net_mean, 0.0, float(self.config.rul_cap))

        # Law of total variance for a two-component mixture: the weighted mean of
        # the component variances, plus the weighted spread of their means about
        # the mixture mean. The second term is what makes the interval widen when
        # the two models disagree.
        within = w * forest_std**2 + (1 - w) * net_std**2
        between = w * (forest_mean - mean) ** 2 + (1 - w) * (net_mean - mean) ** 2
        std = np.sqrt(np.maximum(within + between, 0.0))

        probability = np.clip(w * forest_p + (1 - w) * net_p, 0.0, 1.0)
        return mean.astype("float64"), std.astype("float64"), probability.astype("float64")

    def predict(self, window: NDArray[np.floating]) -> RULPrediction:
        """Predict from a single window of shape ``[T, C]``."""
        array = np.asarray(window)
        if array.ndim != 2:
            raise ValueError(f"expected a single window [T, C], got {array.shape}")
        return self.predict_batch(array[None, ...])[0]

    def predict_batch(self, windows: NDArray[np.floating]) -> list[RULPrediction]:
        """Predict from a batch of shape ``[N, T, C]``."""
        mean, std, probability = self.predict_arrays(windows)
        return [
            RULPrediction(
                rul_mean=float(m),
                rul_std=float(s),
                p_fail_within_h=float(p),
                horizon_h=self.config.horizon_h,
            )
            for m, s, p in zip(mean, std, probability, strict=True)
        ]


class A1Adapter:
    """Wraps a trained A1 network in the ``predict_arrays`` shape the blend expects."""

    def __init__(
        self,
        model: object,
        rul_cap: int = 125,
        horizon_h: int = 30,
        mc_passes: int = 30,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.rul_cap = rul_cap
        self.horizon_h = horizon_h
        self.mc_passes = mc_passes
        self.device = device

    def predict_arrays(
        self, windows: NDArray[np.floating]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Deterministic mean, MC-dropout std, and failure probability."""
        import torch

        from twinmind.models.a1_cnn_bilstm import predict_with_uncertainty

        x = torch.from_numpy(np.asarray(windows, dtype="float32")).to(torch.device(self.device))
        mean, std, probability, _ = predict_with_uncertainty(
            self.model,  # type: ignore[arg-type]
            x,
            n_passes=self.mc_passes,
            rul_cap=float(self.rul_cap),
        )
        return mean, std, probability

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "cpu") -> A1Adapter:
        """Load an A1 checkpoint written by ``train_a1``."""
        import torch

        from twinmind.models.a1_cnn_bilstm import A1Config, A1Net

        payload = torch.load(path, map_location=device, weights_only=False)
        model = A1Net(A1Config(**payload["a1_config"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return cls(model, rul_cap=payload["a1_config"].get("rul_cap", 125), device=device)
