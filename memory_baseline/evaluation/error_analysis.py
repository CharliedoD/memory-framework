from __future__ import annotations

from pathlib import Path
from typing import Any

from memory_baseline.core.io import read_predictions
from memory_baseline.data.longmemeval import is_abstention_sample
from memory_baseline.core.schemas import LongMemEvalSample
from memory_baseline.core.utils import id_key, read_jsonl, write_jsonl


def load_autoeval_labels(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    labels = {}
    for record in read_jsonl(path):
        if "question_id" in record and "autoeval_label" in record:
            labels[str(record["question_id"])] = record["autoeval_label"]
        elif "question_id" in record and "label" in record:
            labels[str(record["question_id"])] = {
                "model": record.get("model"),
                "label": str(record.get("label", "")).upper() == "CORRECT",
            }
    return labels


def write_error_analysis(
    samples: list[LongMemEvalSample],
    retrieval_results: list[dict[str, Any]],
    predictions_path: str | Path,
    token_stats: dict[str, Any],
    output_path: str | Path,
    autoeval_log: str | Path | None = None,
) -> None:
    predictions = read_predictions(predictions_path) if Path(predictions_path).exists() else {}
    labels = load_autoeval_labels(autoeval_log)
    retrieval_by_id = {record["question_id"]: record for record in retrieval_results}
    selected_ids = set(retrieval_by_id) | set(predictions) | set(labels)
    rows = []
    for sample in samples:
        if selected_ids and sample.question_id not in selected_ids:
            continue
        result = retrieval_by_id.get(sample.question_id, {})
        metrics = result.get("metrics", {})
        matched = result.get("matched_turns", [])
        per_question = token_stats.get("per_question", {}).get(sample.question_id, {})
        label = labels.get(sample.question_id)
        recalled_session_ids = sorted({id_key(turn.get("session_id")) for turn in matched})
        recalled_turn_ids = [turn.get("stable_turn_id") for turn in matched]
        top_scores = [turn.get("score") for turn in matched[:5]]
        row = {
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "question": sample.question,
            "gold_answer": sample.answer,
            "hypothesis": predictions.get(sample.question_id, ""),
            "autoeval_label": label,
            "retrieval_session_hit": metrics.get("session_recall_at_k"),
            "retrieval_turn_hit": metrics.get("turn_recall_at_k"),
            "expanded_evidence_hit": metrics.get("expanded_turn_recall_at_k"),
            "answer_session_ids": sample.answer_session_ids,
            "recalled_session_ids": recalled_session_ids,
            "recalled_turn_ids": recalled_turn_ids,
            "top_scores": top_scores,
            "evidence_token_count": metrics.get("evidence_token_count"),
            "build_tokens": per_question.get("build_tokens", 0),
            "query_tokens": per_question.get("query_tokens", 0),
            "likely_failure_type": likely_failure_type(sample, metrics, label),
        }
        rows.append(row)
    write_jsonl(output_path, rows)


def write_error_analysis_from_retrieval_results(
    retrieval_results: list[dict[str, Any]],
    predictions_path: str | Path,
    token_stats: dict[str, Any],
    output_path: str | Path,
    autoeval_log: str | Path | None = None,
) -> None:
    predictions = read_predictions(predictions_path) if Path(predictions_path).exists() else {}
    labels = load_autoeval_labels(autoeval_log)
    rows = []
    for result in retrieval_results:
        question_id = str(result["question_id"])
        matched = result.get("matched_turns", [])
        metrics = result.get("metrics", {})
        per_question = token_stats.get("per_question", {}).get(question_id, {})
        label = labels.get(question_id)
        rows.append(
            {
                "question_id": question_id,
                "question_type": result.get("question_type", ""),
                "question": result.get("question", ""),
                "gold_answer": result.get("answer", ""),
                "hypothesis": predictions.get(question_id, ""),
                "autoeval_label": label,
                "retrieval_session_hit": metrics.get("session_recall_at_k"),
                "retrieval_turn_hit": metrics.get("turn_recall_at_k"),
                "expanded_evidence_hit": metrics.get("expanded_turn_recall_at_k"),
                "answer_session_ids": result.get("answer_session_ids", []),
                "recalled_session_ids": sorted({id_key(turn.get("session_id")) for turn in matched}),
                "recalled_turn_ids": [turn.get("stable_turn_id") for turn in matched],
                "top_scores": [turn.get("score") for turn in matched[:5]],
                "evidence_token_count": metrics.get("evidence_token_count"),
                "build_tokens": per_question.get("build_tokens", 0),
                "query_tokens": per_question.get("query_tokens", 0),
                "likely_failure_type": likely_failure_type_from_fields(
                    question_id,
                    str(result.get("question_type", "")),
                    result.get("answer_session_ids", []),
                    metrics,
                    label,
                ),
            }
        )
    write_jsonl(output_path, rows)


def likely_failure_type(sample: LongMemEvalSample, metrics: dict[str, Any], autoeval_label: Any) -> str:
    return likely_failure_type_from_fields(
        sample.question_id,
        sample.question_type,
        sample.answer_session_ids,
        metrics,
        autoeval_label,
    )


def likely_failure_type_from_fields(
    question_id: str,
    question_type: str,
    answer_session_ids: list[Any],
    metrics: dict[str, Any],
    autoeval_label: Any,
) -> str:
    if is_abstention_sample({"question_id": question_id, "answer_session_ids": answer_session_ids}):
        return "abstention_case"
    evidence_hit = bool(metrics.get("session_recall_at_k") or metrics.get("turn_recall_at_k") or metrics.get("expanded_turn_recall_at_k"))
    if not evidence_hit:
        return "retrieval_miss"
    if _label_is_wrong(autoeval_label):
        if question_type == "temporal-reasoning":
            return "possible_temporal_failure"
        return "answer_failure"
    return "unknown"


def _label_is_wrong(label: Any) -> bool:
    if label is None:
        return False
    if isinstance(label, dict):
        if "label" in label:
            return _label_is_wrong(label["label"])
        return False
    if isinstance(label, bool):
        return not label
    if isinstance(label, (int, float)):
        return float(label) == 0.0
    value = str(label).strip().lower()
    return value in {"wrong", "incorrect", "false", "0", "no"}
