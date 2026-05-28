from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from memory_baseline.core.env import load_project_env
from memory_baseline.core.io import read_predictions
from memory_baseline.core.token_accounting import add_judge_time, add_judge_tokens, new_token_summary
from memory_baseline.core.utils import ensure_dir, read_jsonl, write_json, write_jsonl
from memory_baseline.data.longmemeval import load_samples
from memory_baseline.evaluation.error_analysis import write_error_analysis
from memory_baseline.generation.judge import make_judge


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    samples = load_samples(args.data)
    sample_by_id = {sample.question_id: sample for sample in samples}
    predictions = read_predictions(run_dir / "predictions.jsonl")
    judge = make_judge(args.judge_model, args.judge_base_url)
    existing = {row["question_id"]: row for row in read_jsonl(run_dir / "judge_logs.jsonl")} if args.resume else {}
    pending = [qid for qid in predictions if qid in sample_by_id and qid not in existing]
    log_path = run_dir / "judge_logs.jsonl"
    eval_path = run_dir / f"predictions.jsonl.eval-results-{judge.model_name}"
    if not args.resume:
        log_path.unlink(missing_ok=True)
        eval_path.unlink(missing_ok=True)

    def judge_one(qid: str) -> dict[str, Any]:
        sample = sample_by_id[qid]
        result = judge.judge_longmemeval(
            sample.question_type,
            sample.question,
            sample.answer,
            predictions[qid],
            abstention="_abs" in qid,
        )
        return {
            "question_id": qid,
            "question_type": sample.question_type,
            "model": result.model,
            "label": result.label,
            "score": result.score,
            "raw_response": result.raw_response,
            "judge_input_tokens": result.prompt_tokens,
            "judge_output_tokens": result.completion_tokens,
            "judge_total_tokens": result.total_tokens,
            "provider_usage": result.provider_usage,
            "latency_seconds": result.latency_seconds,
        }

    logs = dict(existing)
    if args.parallelism > 1:
        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            futures = {executor.submit(judge_one, qid): qid for qid in pending}
            for future in as_completed(futures):
                row = future.result()
                logs[row["question_id"]] = row
                _append_jsonl_record(log_path, row)
                _append_jsonl_record(eval_path, _eval_row(sample_by_id[row["question_id"]], predictions[row["question_id"]], row))
    else:
        for qid in pending:
            row = judge_one(qid)
            logs[row["question_id"]] = row
            _append_jsonl_record(log_path, row)
            _append_jsonl_record(eval_path, _eval_row(sample_by_id[row["question_id"]], predictions[row["question_id"]], row))

    ordered_logs = [logs[qid] for qid in predictions if qid in logs]
    if existing and not pending:
        write_jsonl(log_path, ordered_logs)
        write_jsonl(eval_path, [_eval_row(sample_by_id[row["question_id"]], predictions[row["question_id"]], row) for row in ordered_logs])

    metrics = _metrics(ordered_logs)
    retrieval_metrics = _read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    if retrieval_metrics and "session_recall_at_k" in retrieval_metrics:
        write_json(run_dir / "retrieval_metrics.json", retrieval_metrics)
    write_json(run_dir / "qa_metrics.json", metrics)
    write_json(run_dir / "metrics.json", {"retrieval": retrieval_metrics, "qa": metrics})

    token_stats = _read_json(run_dir / "token_stats.json") if (run_dir / "token_stats.json").exists() else new_token_summary()
    token_stats["judge_tokens"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    token_stats.setdefault("time_stats", {})["judge_seconds_excluded"] = 0.0
    token_stats["method_cost_tokens"]["judge_total_tokens_excluded"] = 0
    for row in ordered_logs:
        add_judge_tokens(token_stats, row["question_id"], row.get("judge_input_tokens", 0), row.get("judge_output_tokens", 0))
        add_judge_time(token_stats, row["question_id"], row.get("latency_seconds", 0.0))
    write_json(run_dir / "token_stats.json", token_stats)

    if (run_dir / "retrieval_results.jsonl").exists():
        write_error_analysis(
            samples,
            read_jsonl(run_dir / "retrieval_results.jsonl"),
            run_dir / "predictions.jsonl",
            token_stats,
            run_dir / "error_analysis.jsonl",
            autoeval_log=run_dir / f"predictions.jsonl.eval-results-{judge.model_name}",
        )
    print(f"accuracy={metrics['overall_accuracy']:.4f} correct={metrics['correct']} total={metrics['total']} run_dir={run_dir}")


def _eval_row(sample: Any, hypothesis: str, judge_log: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": sample.question_id,
        "hypothesis": hypothesis,
        "autoeval_label": {
            "model": judge_log["model"],
            "label": judge_log["label"] == "CORRECT",
        },
    }


def _metrics(logs: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in logs:
        by_type[row.get("question_type", "")].append(float(row.get("score", 0.0)))
    correct = sum(1 for row in logs if row.get("label") == "CORRECT")
    total = len(logs)
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "accuracy_by_question_type": {
            question_type: {
                "accuracy": sum(values) / len(values) if values else 0.0,
                "correct": int(sum(values)),
                "total": len(values),
            }
            for question_type, values in sorted(by_type.items())
        },
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


if __name__ == "__main__":
    main()
