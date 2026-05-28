from __future__ import annotations

from typing import Any


def new_token_summary() -> dict[str, Any]:
    return {
        "build_tokens": {
            "embedding_input_tokens": 0,
            "embedding_provider_tokens": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "llm_total_tokens": 0,
        },
        "retrieval_embedding_tokens": {
            "input_tokens": 0,
        },
        "query_tokens": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "judge_tokens": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "method_cost_tokens": {
            "build_embedding_input_tokens": 0,
            "query_total_tokens": 0,
            "judge_total_tokens_excluded": 0,
        },
        "time_stats": {
            "build_seconds": 0.0,
            "query_seconds": 0.0,
            "judge_seconds_excluded": 0.0,
        },
        "per_question": {},
    }


def add_build_tokens(summary: dict[str, Any], question_id: str, input_tokens: int, provider_tokens: int = 0) -> None:
    summary["build_tokens"]["embedding_input_tokens"] += int(input_tokens)
    summary["build_tokens"]["embedding_provider_tokens"] += int(provider_tokens)
    summary["method_cost_tokens"]["build_embedding_input_tokens"] += int(input_tokens)
    per_question = summary["per_question"].setdefault(question_id, {})
    per_question["build_tokens"] = int(input_tokens)


def add_build_time(summary: dict[str, Any], question_id: str, seconds: float) -> None:
    summary.setdefault("time_stats", {}).setdefault("build_seconds", 0.0)
    summary["time_stats"]["build_seconds"] += float(seconds)
    per_question = summary["per_question"].setdefault(question_id, {})
    per_question["build_time_seconds"] = float(seconds)


def add_retrieval_embedding_tokens(summary: dict[str, Any], question_id: str, input_tokens: int) -> None:
    summary["retrieval_embedding_tokens"]["input_tokens"] += int(input_tokens)
    per_question = summary["per_question"].setdefault(question_id, {})
    per_question["retrieval_embedding_tokens"] = int(input_tokens)


def add_query_tokens(summary: dict[str, Any], question_id: str, input_tokens: int, output_tokens: int) -> None:
    total = int(input_tokens) + int(output_tokens)
    summary["query_tokens"]["input_tokens"] += int(input_tokens)
    summary["query_tokens"]["output_tokens"] += int(output_tokens)
    summary["query_tokens"]["total_tokens"] += total
    summary["method_cost_tokens"]["query_total_tokens"] += total
    per_question = summary["per_question"].setdefault(question_id, {})
    per_question["query_tokens"] = total
    per_question["query_input_tokens"] = int(input_tokens)
    per_question["query_output_tokens"] = int(output_tokens)


def add_query_time(summary: dict[str, Any], question_id: str, seconds: float) -> None:
    summary.setdefault("time_stats", {}).setdefault("query_seconds", 0.0)
    summary["time_stats"]["query_seconds"] += float(seconds)
    per_question = summary["per_question"].setdefault(question_id, {})
    per_question["query_time_seconds"] = float(seconds)


def add_judge_tokens(summary: dict[str, Any], question_id: str, input_tokens: int, output_tokens: int) -> None:
    total = int(input_tokens) + int(output_tokens)
    summary["judge_tokens"]["input_tokens"] += int(input_tokens)
    summary["judge_tokens"]["output_tokens"] += int(output_tokens)
    summary["judge_tokens"]["total_tokens"] += total
    summary["method_cost_tokens"]["judge_total_tokens_excluded"] += total
    per_question = summary["per_question"].setdefault(question_id, {})
    per_question["judge_tokens"] = total


def add_judge_time(summary: dict[str, Any], question_id: str, seconds: float) -> None:
    summary.setdefault("time_stats", {}).setdefault("judge_seconds_excluded", 0.0)
    summary["time_stats"]["judge_seconds_excluded"] += float(seconds)
    per_question = summary["per_question"].setdefault(question_id, {})
    per_question["judge_time_seconds"] = float(seconds)
