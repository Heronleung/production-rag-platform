"""RAG evaluation datasets, metrics, runners, and regression gates."""

from evaluation.dataset import dataset_sha256, load_dataset
from evaluation.metrics import aggregate_results, score_retrieval
from evaluation.regression import compare_reports
from evaluation.schema import EvaluationReport, EvaluationResult, GoldenSample

__all__ = [
    "EvaluationReport",
    "EvaluationResult",
    "GoldenSample",
    "aggregate_results",
    "compare_reports",
    "dataset_sha256",
    "load_dataset",
    "score_retrieval",
]
