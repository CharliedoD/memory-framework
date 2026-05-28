from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memory_baseline.cli.run_locomo import write_metrics, write_topk_summary
from memory_baseline.core.utils import ensure_dir, read_jsonl, write_json, write_jsonl
from memory_baseline.data.locomo import parse_int_list


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
    parser.add_argument("--categories", default="1,2,3,4")
    return parser.parse_args(argv)


def merge_runs(run_dirs: list[str | Path], output_dir: str | Path, run_id: str, categories: list[int]) -> Path:
    sources = [Path(path) for path in run_dirs]
    merged_dir = ensure_dir(Path(output_dir) / run_id)
    _merge_flat_runs(sources, merged_dir, categories)
    topk_names = sorted(
        {child.name for source in sources for child in source.glob("top_*") if child.is_dir()},
        key=lambda name: int(name.split("_", 1)[1]),
    )
    for name in topk_names:
        _merge_flat_runs([source / name for source in sources if (source / name).exists()], ensure_dir(merged_dir / name), categories)
    if topk_names:
        write_topk_summary(merged_dir, [int(name.split("_", 1)[1]) for name in topk_names])
    metrics = json.loads((merged_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["merged_from"] = [str(path) for path in sources]
    write_json(merged_dir / "metrics.json", metrics)
    topk_path = merged_dir / "topk_metrics.json"
    if topk_path.exists():
        topk_metrics = json.loads(topk_path.read_text(encoding="utf-8"))
        topk_metrics["merged_from"] = [str(path) for path in sources]
        write_json(topk_path, topk_metrics)
    return merged_dir


def _merge_flat_runs(sources: list[Path], merged_dir: Path, categories: list[int]) -> None:
    for filename in JSONL_FILES:
        rows = []
        seen_question_ids = set()
        for source in sources:
            source_rows = read_jsonl(source / filename)
            for row in source_rows:
                if filename != "build_stats.jsonl" and "question_id" in row:
                    question_id = row["question_id"]
                    if question_id in seen_question_ids:
                        raise ValueError(f"Duplicate question_id {question_id!r} in {filename}")
                    seen_question_ids.add(question_id)
                rows.append(row)
        write_jsonl(merged_dir / filename, rows)

    write_json(merged_dir / "token_stats.json", _merge_token_stats(sources))
    write_json(
        merged_dir / "config.json",
        {
            "merged_from": [str(path) for path in sources],
            "categories": categories,
            "num_source_runs": len(sources),
        },
    )
    write_metrics(merged_dir, categories)


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
        stats_path = source / "token_stats.json"
        if not stats_path.exists():
            continue
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    merged_dir = merge_runs(args.run_dirs, args.output_dir, args.run_id, parse_int_list(args.categories, [1, 2, 3, 4]))
    print(f"merged_run_dir={merged_dir}")


if __name__ == "__main__":
    main()
