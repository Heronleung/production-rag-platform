"""Compare evaluation reports and turn quality regressions into an exit code."""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.schema import EvaluationReport

DEFAULT_TOLERANCES = {
    "hit_rate_at_k": 0.0,
    "mrr_at_k": 0.02,
    "source_recall_at_k": 0.02,
    "key_precision_at_k": 0.02,
}


@dataclass(frozen=True)
class RegressionFailure:
    metric: str
    baseline: float
    current: float
    tolerance: float

    @property
    def message(self) -> str:
        return (
            f"{self.metric} regressed: baseline={self.baseline:.6f}, "
            f"current={self.current:.6f}, allowed_drop={self.tolerance:.6f}"
        )


def compare_reports(
    baseline: EvaluationReport,
    current: EvaluationReport,
    tolerances: dict[str, float] | None = None,
) -> list[RegressionFailure]:
    """Return every higher-is-better metric that dropped beyond tolerance."""
    allowed = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    failures: list[RegressionFailure] = []
    for metric, tolerance in allowed.items():
        old = baseline.aggregate.get(metric)
        new = current.aggregate.get(metric)
        # A metric can be absent when that part of the dataset is not annotated.
        # Compare only metrics present in both reports.
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            continue
        if float(new) + tolerance < float(old):
            failures.append(
                RegressionFailure(metric, float(old), float(new), float(tolerance))
            )
    return failures
