from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memory_baseline.cli.judge_longmemeval import _metrics
from memory_baseline.core.utils import ensure_dir, read_jsonl, write_json, write_jsonl
from memory_baseline.evaluation.error_analysis import write_error_analysis_from_retrieval_results
from memory_baseline.evaluation.retrieval import aggregate_retrieval_results


JSONL_FILES = [
    "build_stats.jsonl",
    "retrieval_results.jsonl",
    "answer_logs.jsonl",
    "predictions.jsonl",
    "judge_logs.jsonl",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="runs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sources = [Path(path) for path in args.run_dirs]
    merged_dir = ensure_dir(Path(args.output_dir) / args.run_id)
    for filename in JSONL_FILES:
        rows = _merge_jsonl(sources, filename)
        write_jsonl(merged_dir / filename, rows)
    retrieval_results = read_jsonl(merged_dir / "retrieval_results.jsonl")
    retrieval_metrics, retrieval_by_type = aggregate_retrieval_results(retrieval_results)
    qa_metrics = _metrics(read_jsonl(merged_dir / "judge_logs.jsonl"))
    token_stats = _merge_token_stats(sources)
    write_json(merged_dir / "retrieval_metrics.json", retrieval_metrics)
    write_json(merged_dir / "retrieval_metrics_by_type.json", retrieval_by_type)
    write_json(merged_dir / "qa_metrics.json", qa_metrics)
    write_json(merged_dir / "metrics.json", {"retrieval": retrieval_metrics, "retrieval_by_type": retrieval_by_type, "qa": qa_metrics, "merged_from": [str(path) for path in sources]})
    write_json(merged_dir / "token_stats.json", token_stats)
    write_error_analysis_from_retrieval_results(
        retrieval_results,
        merged_dir / "predictions.jsonl",
        token_stats,
        merged_dir / "error_analysis.jsonl",
        autoeval_log=merged_dir / "judge_logs.jsonl",
    )
    write_json(merged_dir / "config.json", _merge_config(sources, args.run_id))
    print(f"merged_run_dir={merged_dir} accuracy={qa_metrics['overall_accuracy']:.4f} correct={qa_metrics['correct']} total={qa_metrics['total']}")


def _merge_jsonl(sources: list[Path], filename: str, dedupe: bool = False) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for source in sources:
        for row in read_jsonl(source / filename):
            question_id = row.get("question_id")
            if question_id and filename != "build_stats.jsonl":
                if question_id in seen:
                    if dedupe:
                        continue
                    raise ValueError(f"Duplicate question_id {question_id!r} in {filename}")
                seen.add(question_id)
            rows.append(row)
    return rows


def _merge_token_stats(sources: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "build_tokens": {},
        "retrieval_embedding_tokens": {},
        "query_tokens": {},
        "judge_tokens": {},
        "method_cost_tokens": {},
        "time_stats": {},
        "per_question": {},
    }
    for source in sources:
        path = source / "token_stats.json"
        if not path.exists():
            continue
        stats = json.loads(path.read_text(encoding="utf-8"))
        for section in ["build_tokens", "retrieval_embedding_tokens", "query_tokens", "judge_tokens", "method_cost_tokens", "time_stats"]:
            for key, value in stats.get(section, {}).items():
                if isinstance(value, (int, float)):
                    merged[section][key] = merged[section].get(key, 0) + value
                else:
                    merged[section][key] = value
        for key, value in stats.get("per_question", {}).items():
            if key in merged["per_question"]:
                raise ValueError(f"Duplicate token_stats key {key!r}")
            merged["per_question"][key] = value
    return merged


def _merge_config(sources: list[Path], run_id: str) -> dict[str, Any]:
    configs = []
    for source in sources:
        path = source / "config.json"
        if path.exists():
            configs.append(json.loads(path.read_text(encoding="utf-8")))

    merged: dict[str, Any] = {"run_id": run_id, "merged_from": [str(path) for path in sources], "num_source_runs": len(sources)}
    if not configs:
        return merged

    skip = {"run_id", "selected_question_ids", "num_selected_samples"}
    for key, value in configs[0].items():
        if key in skip:
            continue
        values = [config.get(key) for config in configs]
        merged[key] = value if all(item == value for item in values) else values

    merged["source_run_ids"] = [config.get("run_id") for config in configs]
    merged["num_selected_samples"] = sum(int(config.get("num_selected_samples") or 0) for config in configs)
    selected_ids = []
    for config in configs:
        selected_ids.extend(config.get("selected_question_ids") or [])
    if selected_ids:
        merged["selected_question_ids"] = selected_ids
    return merged


if __name__ == "__main__":
    main()
