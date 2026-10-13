"""Model C: a heteroscedastic residual MLP ensemble emulating the twin.

WHAT IT EMULATES. **The twin, not reality.** It inherits every assumption in
``configs/twin.yaml`` -- including ``demand_capacity_ratio``, a 25% perturbation
of which moves twin profit by 45%, and the 0.95 warning threshold chosen to keep
the landscape non-degenerate rather than measured. A surrogate that matches the
twin perfectly is still only as good as the twin.

TWO UNCERTAINTIES, AND WHY THE ENSEMBLE EXISTS.

The NLL head learns the **aleatoric** std: the spread of the label, which is a
mean of ``n_reps`` twin replications, so it is the per-replication noise divided
by ``sqrt(n_reps)``. It answers "how noisy is the twin here". It does **not**
answer "do I know this region", and it cannot: a single network is perfectly
capable of being confidently wrong somewhere it has never seen.

The **epistemic** std is the disagreement between ensemble members, which differ
only in initialisation seed and in a bootstrap of context blocks. Where the data
constrains the function they agree; where it does not they diverge. That is the
only one of the two that should ever trigger a fallback to the real twin, and
the evaluation reports whether it separates the OOD split from T1.

ARCHITECTURE. Four pre-norm residual blocks of width 256 with GELU, then one
small head per KPI emitting a mean and a softplus std with a floor. Pre-norm
rather than post-norm because it trains without warm-up tricks at this depth,
and a floor on the std because an NLL with an unbounded-below std has a
degenerate optimum: drive the std to zero on the easiest points and the loss
goes to minus infinity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from twinmind.models.surrogate import KPI_ORDER, SurrogateOutput
from twinmind.models.surrogate_transforms import TRANSFORMS, inverse_predictions
from twinmind.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from twinmind.twin.context import TwinContext
    from twinmind.twin.runner import Scenario

logger = get_logger(__name__)


@dataclass
class SurrogateConfig:
    """Architecture and training settings, saved beside the weights."""

    input_dim: int
    hidden_dim: int = 256
    n_blocks: int = 4
    dropout: float = 0.05
    #: Floor on the predicted std, in standardised transformed units. Without it
    #: the Gaussian NLL is unbounded below: a model can drive sigma to zero on
    #: the points it fits best and take the loss to minus infinity.
    min_std: float = 1e-3
    n_members: int = 5
    epochs: int = 120
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    patience: int = 15
    bootstrap_blocks: bool = True
    seed: int = 42
    feature_names: list[str] = field(default_factory=list)
    transforms: dict[str, str] = field(default_factory=lambda: dict(TRANSFORMS))


class ResidualBlock(nn.Module):
    """Pre-norm residual block: ``x + W2 GELU(W1 LN(x))``."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One residual step."""
        h = self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))
        return x + h


class HeteroscedasticMLP(nn.Module):
    """Predicts a mean and a std per KPI from the encoded features."""

    def __init__(self, config: SurrogateConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Linear(config.input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualBlock(config.hidden_dim, config.dropout) for _ in range(config.n_blocks)
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        # One head per KPI rather than a single 16-wide output: the KPIs have
        # very different functional shapes (a near-binary stockout flag against
        # a smooth profit surface) and separate heads let each have its own last
        # layer while sharing the trunk that learns the line's dynamics.
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 4),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 4, 2),
            )
            for _ in KPI_ORDER
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mean, std)``, both ``[B, 8]``, in standardised transformed space."""
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        outputs = [head(h) for head in self.heads]
        mean = torch.stack([o[:, 0] for o in outputs], dim=1)
        raw = torch.stack([o[:, 1] for o in outputs], dim=1)
        std = nn.functional.softplus(raw) + self.config.min_std
        return mean, std


def gaussian_nll(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood, mean over all elements.

    ``0.5 * (log(2 pi sigma^2) + ((y - mu)/sigma)^2)``. The constant is kept so
    the reported number is a real NLL comparable across runs rather than an
    unnamed quantity.
    """
    variance = std**2
    return 0.5 * (torch.log(2 * torch.pi * variance) + (target - mean) ** 2 / variance).mean()


@dataclass
class SurrogateEnsemble:
    """The production surrogate: N members, their scalers, and the Protocol.

    Members differ in initialisation seed and, by default, in a bootstrap over
    **context blocks** rather than over rows. Resampling rows would give every
    member nearly the same contexts -- 64 scenarios share each one -- and the
    disagreement would collapse to initialisation noise alone, which understates
    epistemic uncertainty exactly where it matters.
    """

    members: list[HeteroscedasticMLP]
    config: SurrogateConfig
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    state_mean: NDArray[np.float64]
    state_scale: NDArray[np.float64]
    target_mean: NDArray[np.float64]
    target_scale: NDArray[np.float64]

    # ------------------------------------------------------------- inference --
    def predict_features(self, features: NDArray[np.float64]) -> SurrogateOutput:
        """Predict from already-encoded features.

        The ensemble's mean is the average of member means. The **aleatoric**
        std combines the members' own predicted stds in quadrature-average, and
        the **epistemic** std is the spread of their means -- the standard
        decomposition of a Gaussian mixture's variance into within-component and
        between-component parts.
        """
        x = torch.from_numpy(
            (np.asarray(features, dtype="float64") - self.feature_mean) / self.feature_scale
        ).float()

        means, stds = [], []
        for member in self.members:
            member.eval()
            with torch.no_grad():
                mean, std = member(x)
            means.append(mean.numpy().astype("float64"))
            stds.append(std.numpy().astype("float64"))

        stacked_mean = np.stack(means)
        stacked_std = np.stack(stds)

        ensemble_mean = stacked_mean.mean(axis=0)
        aleatoric = np.sqrt((stacked_std**2).mean(axis=0))
        # ddof=0: this is the spread of the members we have, not an estimate of
        # a wider population's spread. With 5 members ddof=1 would inflate it by
        # 12% for no reason.
        epistemic = stacked_mean.std(axis=0, ddof=0)

        # Undo target standardisation, then the per-KPI transform.
        mean_in_space = ensemble_mean * self.target_scale + self.target_mean
        aleatoric_in_space = aleatoric * self.target_scale
        epistemic_in_space = epistemic * self.target_scale

        final_mean, final_aleatoric = inverse_predictions(mean_in_space, aleatoric_in_space)
        _, final_epistemic = inverse_predictions(mean_in_space, epistemic_in_space)
        return SurrogateOutput(
            mean=final_mean,
            aleatoric_std=final_aleatoric,
            epistemic_std=final_epistemic,
        )

    def predict(self, scenarios: Sequence[Scenario], context: TwinContext) -> SurrogateOutput:
        """Predict KPIs for a batch of scenarios at one context."""
        from twinmind.models.surrogate import encode_batch

        features = encode_batch(
            scenarios,
            context,
            state_mean=self.state_mean,
            state_scale=self.state_scale,
        )
        return self.predict_features(features)

    # ------------------------------------------------------------- artifacts --
    def save(self, directory: Path) -> Path:
        """Write members, scalers and the config."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for index, member in enumerate(self.members):
            torch.save(member.state_dict(), directory / f"member_{index}.pt")
        np.savez(
            directory / "scalers.npz",
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            state_mean=self.state_mean,
            state_scale=self.state_scale,
            target_mean=self.target_mean,
            target_scale=self.target_scale,
        )
        (directory / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        logger.info("wrote surrogate ensemble (%d members) to %s", len(self.members), directory)
        return directory

    @classmethod
    def load(cls, directory: Path) -> SurrogateEnsemble:
        """Read an ensemble written by :meth:`save`."""
        directory = Path(directory)
        payload = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        config = SurrogateConfig(**payload)
        members = []
        for path in sorted(directory.glob("member_*.pt")):
            member = HeteroscedasticMLP(config)
            member.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            member.eval()
            members.append(member)
        if not members:
            raise FileNotFoundError(f"no member_*.pt in {directory}")
        with np.load(directory / "scalers.npz") as scalers:
            return cls(
                members=members,
                config=config,
                feature_mean=scalers["feature_mean"],
                feature_scale=scalers["feature_scale"],
                state_mean=scalers["state_mean"],
                state_scale=scalers["state_scale"],
                target_mean=scalers["target_mean"],
                target_scale=scalers["target_scale"],
            )


def train_member(
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
    val_features: NDArray[np.float64],
    val_targets: NDArray[np.float64],
    config: SurrogateConfig,
    seed: int,
) -> tuple[HeteroscedasticMLP, dict[str, Any]]:
    """Train one ensemble member; return it with its best-epoch weights restored.

    Early stopping is on the validation NLL over held-out **context blocks**, so
    "better" means better at a plant state the member has never seen.
    """
    torch.manual_seed(seed)
    model = HeteroscedasticMLP(config)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    x = torch.from_numpy(features).float()
    y = torch.from_numpy(targets).float()
    xv = torch.from_numpy(val_features).float()
    yv = torch.from_numpy(val_targets).float()

    n_batches = max(1, int(np.ceil(len(x) / config.batch_size)))
    total_steps = config.epochs * n_batches
    warmup = max(1, int(config.warmup_fraction * total_steps))

    def lr_at(step: int) -> float:
        """Linear warm-up then cosine decay to a tenth of the peak."""
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at)
    generator = torch.Generator().manual_seed(seed + 7919)

    best_loss, best_state, best_epoch, since_best = float("inf"), None, -1, 0
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(x), config.batch_size):
            batch = order[start : start + config.batch_size]
            mean, std = model(x[batch])
            loss = gaussian_nll(mean, std, y[batch])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            scheduler.step()
            epoch_loss += float(loss) * len(batch)

        model.eval()
        with torch.no_grad():
            vmean, vstd = model(xv)
            val_loss = float(gaussian_nll(vmean, vstd, yv))
        history.append({"epoch": epoch, "train_nll": epoch_loss / len(x), "val_nll": val_loss})

        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, since_best = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_nll": best_loss,
        "epochs_run": len(history),
        "history": history,
    }
