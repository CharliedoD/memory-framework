from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from memory_baseline.core.utils import read_jsonl, write_json


BOOL_METRICS = [
    "session_recall_at_k",
    "turn_recall_at_k",
    "expanded_turn_recall_at_k",
    "turn_or_expanded_recall_at_k",
]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_retrieval_results(results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    valid = [result for result in results if not result.get("metrics", {}).get("skipped")]
    overall = _aggregate_group(valid)
    by_type: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in valid:
        grouped[result.get("question_type", "unknown")].append(result)
    for question_type, group in sorted(grouped.items()):
        by_type[question_type] = _aggregate_group(group)
    overall["num_total"] = len(results)
    overall["num_eval"] = len(valid)
    overall["num_skipped_abstention"] = len(results) - len(valid)
    return overall, by_type


def _aggregate_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [result.get("metrics", {}) for result in results]
    output: dict[str, Any] = {"count": len(results)}
    for name in BOOL_METRICS:
        values = [1.0 if metric.get(name) else 0.0 for metric in metrics if metric.get(name) is not None]
        output[name] = _mean(values)
    output["avg_evidence_tokens"] = _mean([float(metric.get("evidence_token_count", 0)) for metric in metrics])
    output["avg_num_sessions"] = _mean([float(metric.get("num_deduped_sessions", 0)) for metric in metrics])
    output["avg_num_turns"] = _mean([float(metric.get("num_deduped_turns", 0)) for metric in metrics])
    output["avg_num_matched_turns"] = _mean([float(metric.get("num_matched_turns", 0)) for metric in metrics])
    return output


def write_retrieval_metrics(retrieval_results_path: str | Path, metrics_path: str | Path, by_type_path: str | Path) -> None:
    results = read_jsonl(retrieval_results_path)
    overall, by_type = aggregate_retrieval_results(results)
    write_json(metrics_path, overall)
    write_json(by_type_path, by_type)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("retrieval_results")
    parser.add_argument("--metrics-out", default="metrics.json")
    parser.add_argument("--by-type-out", default="metrics_by_type.json")
    args = parser.parse_args(argv)
    write_retrieval_metrics(args.retrieval_results, args.metrics_out, args.by_type_out)


if __name__ == "__main__":
    main()
