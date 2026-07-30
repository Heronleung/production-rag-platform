"""Run a reproducible RAG evaluation and optionally enforce a regression gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from api.embeddings import get_embedder
from api.llm import get_llm
from api.vectorstore.milvus_store import MilvusStore
from evaluation.dataset import dataset_sha256, load_dataset
from evaluation.regression import compare_reports
from evaluation.runner import run_evaluation
from evaluation.schema import EvaluationReport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="evaluation/datasets/golden.jsonl", help="Golden JSONL dataset."
    )
    parser.add_argument(
        "--strategy",
        choices=("vanilla", "mmr", "multi_query", "combined"),
        default="vanilla",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mmr-lambda", type=float, default=0.5)
    parser.add_argument("--multi-query-count", type=int, default=3)
    parser.add_argument("--output", required=True, help="Destination JSON report.")
    parser.add_argument("--baseline", help="Existing report to compare against.")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 1 when a deterministic metric drops beyond its tolerance.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    samples = load_dataset(dataset_path)
    embedder = get_embedder()
    llm = get_llm()
    store = MilvusStore(dim=embedder.dim)

    report = run_evaluation(
        samples,
        strategy=args.strategy,
        dataset_path=str(dataset_path),
        dataset_hash=dataset_sha256(dataset_path),
        embedder=embedder,
        store=store,
        llm=llm,
        top_k=args.top_k,
        mmr_lambda=args.mmr_lambda,
        multi_query_count=args.multi_query_count,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.aggregate, indent=2))
    print(f"report: {output}")

    if args.baseline:
        baseline = EvaluationReport.model_validate_json(Path(args.baseline).read_text())
        failures = compare_reports(baseline, report)
        if failures:
            print("regressions:")
            for failure in failures:
                print(f"- {failure.message}")
            if args.fail_on_regression:
                return 1
        else:
            print("regression gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
