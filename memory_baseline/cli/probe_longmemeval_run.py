from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory_baseline.core.env import load_project_env
from memory_baseline.core.io import read_predictions
from memory_baseline.core.llm_cache import cached_response, write_cached_response
from memory_baseline.core.utils import estimate_messages_tokens, estimate_tokens, read_jsonl, write_json, write_jsonl
from memory_baseline.data.longmemeval import is_abstention_sample, load_samples
from memory_baseline.generation.judge import _open_json_with_retries, make_judge


NO_MEMORY_SYSTEM_PROMPT = """You are answering a question without access to prior conversation memory.
The user's question date is: {question_date}.
If the question requires private conversation history, answer that you do not have enough information.
Keep the answer brief and factual."""


PROBE_SYSTEM_PROMPT = "You diagnose memory-augmented QA failures. Return JSON only."


PROBE_PROMPT = """Question type: {question_type}
Question date: {question_date}
Question: {question}
Gold answer: {gold_answer}

Retrieved evidence:
{evidence}

System answer with retrieved evidence:
{system_answer}
System answer was judged correct: {system_correct}

No-memory answer:
{no_memory_answer}
No-memory answer was judged correct: {no_memory_correct}

Diagnose the case. Use these definitions:
- evidence_sufficient: true only if the retrieved evidence contains enough information to produce the gold answer, or for unanswerable questions, enough information to justify that the requested information is unavailable.
- retrieval_failure: the system answer is wrong and the retrieved evidence is insufficient or misses required details.
- utilization_failure: the system answer is wrong, but sufficient evidence is present and the model failed to read, aggregate, compare, or calculate correctly.
- hallucination: the system answer is wrong and directly contradicts the retrieved evidence.
- correct: the system answer is correct.

Choose primary_architecture_need from:
retrieval_precision, evidence_packing, temporal_reasoning, multi_session_aggregation, answerer_reasoning, abstention_calibration, none.

Return JSON with exactly these keys:
{{
  "evidence_sufficient": true/false,
  "noise_level": "low|medium|high",
  "failure_category": "correct|retrieval_failure|utilization_failure|hallucination",
  "primary_architecture_need": "retrieval_precision|evidence_packing|temporal_reasoning|multi_session_aggregation|answerer_reasoning|abstention_calibration|none",
  "reason": "one short sentence",
  "key_evidence": "short quote or summary of the relevant evidence, or empty string"
}}"""


@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float
    cache_hit: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--probe-model", default=os.getenv("JUDGE_MODEL"))
    parser.add_argument("--probe-base-url", default=os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--answer-model", default=os.getenv("ANSWER_MODEL") or os.getenv("ANSWERER_MODEL") or os.getenv("LLM_MODEL"))
    parser.add_argument("--answer-base-url", default=os.getenv("ANSWER_BASE_URL") or os.getenv("ANSWERER_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "probes"
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.data)
    retrieval_by_id = {row["question_id"]: row for row in read_jsonl(run_dir / "retrieval_results.jsonl")}
    predictions = read_predictions(run_dir / "predictions.jsonl")
    judge_by_id = {row["question_id"]: row for row in read_jsonl(run_dir / "judge_logs.jsonl")}
    existing = _load_existing(output_dir) if args.resume else {}

    judge = make_judge(args.probe_model, args.probe_base_url)
    items = [sample for sample in samples if sample.question_id in retrieval_by_id and sample.question_id in predictions]
    pending = [sample for sample in items if sample.question_id not in existing]
    rows = list(existing.values())

    def process_one(sample: Any) -> dict[str, Any]:
        question_id = sample.question_id
        no_memory = _no_memory_answer(args, sample)
        no_memory_judge = judge.judge_longmemeval(
            sample.question_type,
            sample.question,
            sample.answer,
            no_memory.text,
            abstention=is_abstention_sample(sample),
        )
        system_correct = str(judge_by_id.get(question_id, {}).get("label", "")).upper() == "CORRECT"
        no_memory_correct = no_memory_judge.label == "CORRECT"
        probe = _probe_case(
            args,
            sample,
            retrieval_by_id[question_id],
            predictions[question_id],
            system_correct,
            no_memory.text,
            no_memory_correct,
        )
        probe_input_tokens = probe.pop("_probe_input_tokens")
        probe_output_tokens = probe.pop("_probe_output_tokens")
        probe_total_tokens = probe.pop("_probe_total_tokens")
        return {
            "question_id": question_id,
            "question_type": sample.question_type,
            "question": sample.question,
            "gold_answer": sample.answer,
            "is_abstention": is_abstention_sample(sample),
            "system_answer": predictions[question_id],
            "system_correct": system_correct,
            "no_memory_answer": no_memory.text,
            "no_memory_correct": no_memory_correct,
            "retrieval_session_hit": retrieval_by_id[question_id].get("metrics", {}).get("session_recall_at_k"),
            "retrieval_turn_hit": retrieval_by_id[question_id].get("metrics", {}).get("turn_recall_at_k"),
            "expanded_evidence_hit": retrieval_by_id[question_id].get("metrics", {}).get("expanded_turn_recall_at_k"),
            "evidence_token_count": retrieval_by_id[question_id].get("metrics", {}).get("evidence_token_count"),
            **probe,
            "no_memory_input_tokens": no_memory.input_tokens,
            "no_memory_output_tokens": no_memory.output_tokens,
            "no_memory_total_tokens": no_memory.total_tokens,
            "no_memory_judge_tokens": no_memory_judge.total_tokens,
            "probe_input_tokens": probe_input_tokens,
            "probe_output_tokens": probe_output_tokens,
            "probe_total_tokens": probe_total_tokens,
        }

    if pending:
        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            futures = [executor.submit(process_one, sample) for sample in pending]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                _append_jsonl(output_dir / "probe_cases.jsonl", row)
    elif rows:
        write_jsonl(output_dir / "probe_cases.jsonl", rows)

    rows.sort(key=lambda row: [sample.question_id for sample in items].index(row["question_id"]))
    summary = _summarize(rows)
    write_json(
        output_dir / "probe_config.json",
        {
            "data": args.data,
            "run_dir": str(run_dir),
            "probe_model": args.probe_model,
            "answer_model": args.answer_model,
            "parallelism": args.parallelism,
            "num_cases": len(rows),
        },
    )
    write_json(output_dir / "probe_summary.json", summary)
    (output_dir / "probe_report.md").write_text(_render_report(summary), encoding="utf-8")
    print(f"probe_dir={output_dir} num_cases={len(rows)}")


def _load_existing(output_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        row["question_id"]: row
        for row in read_jsonl(output_dir / "probe_cases.jsonl")
        if "question_id" in row
    }


def _no_memory_answer(args: argparse.Namespace, sample: Any) -> ChatResult:
    messages = [
        {"role": "system", "content": NO_MEMORY_SYSTEM_PROMPT.format(question_date=sample.question_date)},
        {"role": "user", "content": f"Question: {sample.question}\n\nExpected answer format: concise natural language."},
    ]
    return _chat(args.answer_model, args.answer_base_url, "answer_no_memory", messages)


def _probe_case(
    args: argparse.Namespace,
    sample: Any,
    retrieval: dict[str, Any],
    system_answer: str,
    system_correct: bool,
    no_memory_answer: str,
    no_memory_correct: bool,
) -> dict[str, Any]:
    prompt = PROBE_PROMPT.format(
        question_type=sample.question_type,
        question_date=sample.question_date,
        question=sample.question,
        gold_answer=sample.answer,
        evidence=retrieval.get("formatted_evidence", ""),
        system_answer=system_answer,
        system_correct=json.dumps(system_correct),
        no_memory_answer=no_memory_answer,
        no_memory_correct=json.dumps(no_memory_correct),
    )
    messages = [{"role": "system", "content": PROBE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    result = _chat(
        args.probe_model,
        args.probe_base_url,
        "probe",
        messages,
        max_tokens=512,
    )
    parsed = _parse_json(result.text)
    if not parsed:
        result = _chat(
            args.probe_model,
            args.probe_base_url,
            "probe_retry",
            messages,
            max_tokens=768,
        )
        parsed = _parse_json(result.text)
    category = str(parsed.get("failure_category", "")).strip()
    if system_correct:
        category = "correct"
    if category not in {"correct", "retrieval_failure", "utilization_failure", "hallucination"}:
        category = "utilization_failure" if parsed.get("evidence_sufficient") else "retrieval_failure"
    need = str(parsed.get("primary_architecture_need", "")).strip()
    if need not in {
        "retrieval_precision",
        "evidence_packing",
        "temporal_reasoning",
        "multi_session_aggregation",
        "answerer_reasoning",
        "abstention_calibration",
        "none",
    }:
        need = "none" if category == "correct" else "answerer_reasoning"
    return {
        "evidence_sufficient": bool(parsed.get("evidence_sufficient")),
        "noise_level": str(parsed.get("noise_level", "medium")).strip(),
        "failure_category": category,
        "primary_architecture_need": need,
        "reason": str(parsed.get("reason", "")),
        "key_evidence": str(parsed.get("key_evidence", "")),
        "raw_probe_response": result.text,
        "_probe_input_tokens": result.input_tokens,
        "_probe_output_tokens": result.output_tokens,
        "_probe_total_tokens": result.total_tokens,
    }


def _chat(
    model: str | None,
    base_url: str | None,
    kind: str,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
) -> ChatResult:
    if not model:
        raise RuntimeError("model is required")
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("ANSWER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("API key is required")
    payload_obj: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
    if max_tokens is not None:
        payload_obj["max_tokens"] = max_tokens
    cached = cached_response(kind, model, payload_obj)
    if cached is not None:
        return _chat_result(cached, messages, model, 0.0, True)
    payload = json.dumps(payload_obj).encode("utf-8")
    request = urllib.request.Request(
        f"{(base_url or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1').rstrip('/')}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    body = _open_json_with_retries(request)
    latency = time.time() - started
    write_cached_response(kind, model, payload_obj, body)
    return _chat_result(body, messages, model, latency, False)


def _chat_result(body: dict[str, Any], messages: list[dict[str, str]], model: str, latency: float, cache_hit: bool) -> ChatResult:
    text = body["choices"][0].get("message", {}).get("content", "").strip()
    usage = dict(body.get("usage", {}))
    input_tokens = int(usage.get("prompt_tokens") or estimate_messages_tokens(messages))
    output_tokens = int(usage.get("completion_tokens") or estimate_tokens(text))
    return ChatResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=int(usage.get("total_tokens") or input_tokens + output_tokens),
        latency_seconds=latency,
        cache_hit=cache_hit,
    )


def _parse_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = _summarize_group(rows)
    by_type: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["question_type"]].append(row)
    for question_type, group in sorted(grouped.items()):
        by_type[question_type] = _summarize_group(group)
    return {"overall": overall, "by_type": by_type}


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = [row for row in rows if not row["system_correct"]]
    return {
        "total": len(rows),
        "system_correct": sum(1 for row in rows if row["system_correct"]),
        "no_memory_correct": sum(1 for row in rows if row["no_memory_correct"]),
        "memory_beneficial": sum(1 for row in rows if row["system_correct"] and not row["no_memory_correct"]),
        "memory_harmful": sum(1 for row in rows if not row["system_correct"] and row["no_memory_correct"]),
        "both_correct": sum(1 for row in rows if row["system_correct"] and row["no_memory_correct"]),
        "both_wrong": sum(1 for row in rows if not row["system_correct"] and not row["no_memory_correct"]),
        "wrong_total": len(wrong),
        "wrong_failure_categories": dict(Counter(row["failure_category"] for row in wrong)),
        "wrong_architecture_needs": dict(Counter(row["primary_architecture_need"] for row in wrong)),
        "wrong_noise_levels": dict(Counter(row["noise_level"] for row in wrong)),
        "wrong_evidence_sufficient": sum(1 for row in wrong if row["evidence_sufficient"]),
        "avg_evidence_tokens": _mean([float(row.get("evidence_token_count") or 0) for row in rows]),
        "probe_total_tokens": sum(int(row.get("probe_total_tokens") or 0) for row in rows),
        "no_memory_total_tokens": sum(int(row.get("no_memory_total_tokens") or 0) for row in rows),
        "no_memory_judge_tokens": sum(int(row.get("no_memory_judge_tokens") or 0) for row in rows),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _render_report(summary: dict[str, Any]) -> str:
    lines = ["# LongMemEval Probe Report", ""]
    overall = summary["overall"]
    lines.extend(
        [
            "## Overall",
            "",
            f"- System correct: {overall['system_correct']} / {overall['total']}",
            f"- No-memory correct: {overall['no_memory_correct']} / {overall['total']}",
            f"- Memory beneficial: {overall['memory_beneficial']}",
            f"- Memory harmful: {overall['memory_harmful']}",
            f"- Wrong failure categories: {overall['wrong_failure_categories']}",
            f"- Wrong architecture needs: {overall['wrong_architecture_needs']}",
            f"- Wrong evidence sufficient: {overall['wrong_evidence_sufficient']} / {overall['wrong_total']}",
            "",
            "## By Type",
            "",
            "| Type | Correct | No-memory | Beneficial | Harmful | Wrong failures | Architecture needs |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for question_type, row in summary["by_type"].items():
        lines.append(
            f"| `{question_type}` | {row['system_correct']}/{row['total']} | "
            f"{row['no_memory_correct']}/{row['total']} | {row['memory_beneficial']} | {row['memory_harmful']} | "
            f"{row['wrong_failure_categories']} | {row['wrong_architecture_needs']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
