"""Do the agent's explanations match the numbers it was given?

Checks the Explainer's prose against the deterministic KPI record: every figure
cited must appear in the result, and the claimed direction of every effect must
match the computed sign.
"""

from __future__ import annotations

from twinmind.twin.runner import TwinResult


def extract_claims(explanation: str) -> list[dict[str, object]]:
    """Pull numeric claims out of the explanation text."""
    raise NotImplementedError("extract_claims is not implemented yet")


def verify_claims(claims: list[dict[str, object]], result: TwinResult, tolerance: float = 0.01
                  ) -> list[dict[str, object]]:
    """Check each claim against the deterministic result."""
    raise NotImplementedError("verify_claims is not implemented yet")


def faithfulness_score(explanation: str, result: TwinResult) -> float:
    """Fraction of numeric claims that survive verification."""
    raise NotImplementedError("faithfulness_score is not implemented yet")
