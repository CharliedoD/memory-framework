from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from memory_baseline.core.env import load_project_env
from memory_baseline.core.io import read_predictions, write_predictions
from memory_baseline.core.llm_cache import cached_response, write_cached_response
from memory_baseline.core.token_accounting import add_judge_time, add_judge_tokens, add_query_time, add_query_tokens, new_token_summary
from memory_baseline.core.utils import ensure_dir, estimate_messages_tokens, estimate_tokens, read_jsonl, write_json, write_jsonl
from memory_baseline.data.longmemeval import is_abstention_sample, load_question_ids, load_samples
from memory_baseline.evaluation.error_analysis import write_error_analysis
from memory_baseline.generation.answerer import build_answer_messages as build_baseline_answer_messages
from memory_baseline.generation.judge import get_longmemeval_prompt
from memory_baseline.retrieval.formatter import format_evidence_for_answerer


ANSWER_SYSTEM_PROMPT = """You answer LongMemEval memory questions using only recalled memory evidence.
The user's question date is: {question_date}.
Interpret today, yesterday, last week, current, now, and other relative dates against the question date.
Think through the evidence internally, then output only the final answer.
If the evidence is insufficient, answer exactly: The information is not available in the recalled memory."""


ANSWER_INSTRUCTIONS = {
    "multi-session": "Combine all relevant sessions before answering. For counts, lists, sums, or comparisons, deduplicate repeated mentions and answer from the complete set.",
    "temporal-reasoning": "Order relevant memories by timestamp before answering. For elapsed-time questions, compute against the question date.",
    "knowledge-update": "If memories conflict because information was updated, use the latest relevant update unless the question asks about an earlier state.",
    "single-session-preference": "Use the recalled personal preference or constraint directly; do not give generic advice when a specific preference is available.",
}


MEM0_LME_ANSWER_PROMPT = """You are a personal assistant with access to memories from past conversations with a user. Answer the question using information from the recalled memories below.

IMPORTANT: Today's date is {question_date}. Compute relative time expressions against this date.
IMPORTANT: If the memories contain the numbers, dates, prices, ages, or facts needed to answer, do the computation. Do not abstain when the raw data exists, even if it is scattered across different conversations.
IMPORTANT: Pay attention to the exact entity, role, title, variant, and context in the question. If memories only mention a different entity or context, say the information is not available.
IMPORTANT: Use only the recalled memories. Do not invent numbers, prices, dates, addresses, names, or events.

Before answering, reason step-by-step inside <mem_thinking> tags:
- List every relevant memory, including memories that appear late in the evidence.
- For counting, list, sum, or comparison questions: enumerate every candidate item with date/session context, deduplicate repeated mentions, then compute the final answer.
- For cross-topic computation: identify each needed fact independently and where it appears, then compute.
- For temporal questions: identify the relevant event dates and compute intervals from the question date or from the event reference point asked by the question.
- For knowledge updates or conflicting values: use the latest relevant update unless the question asks about an earlier state.
- For preference or recommendation questions: list the user's relevant preferences, constraints, owned tools/resources, and anti-preferences before selecting the answer.
- Before abstaining, check whether any memory contains a relevant direct or indirect fact. Abstain only when the requested information is genuinely missing or the question asks about a mismatched entity/context.

Rules:
1. Always try to answer if the topic appears in memory, even indirectly.
2. Most recent wins for current status, counts, preferences, or updated facts. Historical event dates should use the memory closest to the event when applicable.
3. For time-bounded questions, compute the date window first, then scan all memories for events in range.
4. For comparison or ordering questions, both compared items must be supported by memory.
5. Actions and completed events beat plans. If a plan has a specific date and no later contradiction, treat it as likely completed on that date.
6. User-stated facts are stronger evidence for personal questions than assistant advice or generic information.
7. Keep the final answer concise and specific.

{evidence_text}

Question: {question}

Work through the instructions in <mem_thinking> tags, then give the final answer after "ANSWER:"."""


GLOBAL_REASONING_ANSWER_PROMPT = """You answer LongMemEval memory questions using only recalled memory evidence.

The user's question date is {question_date}. Interpret relative time expressions against this date.
Use only the recalled memories and the current question. Do not invent unsupported names, numbers, dates, prices, preferences, or events.

Before answering, reason step-by-step inside <mem_thinking> tags:
- Identify what kind of answer the question asks for: direct fact, list, count, comparison, temporal calculation, update/current state, or personalized recommendation.
- List every recalled memory that is relevant to that answer. Do not stop after the first match.
- If the question asks for a personalized recommendation or preference-based response, extract the user's relevant preferences, anti-preferences, constraints, owned tools/resources, and prior interests from the memories. Then tailor the answer to those details and avoid generic suggestions when specific preferences exist.
- If the question asks for a count, list, sum, or comparison, enumerate each candidate item with date/session context, deduplicate repeated mentions, then compute the answer.
- If the question asks about time, identify the relevant dates and compute against the question date or the reference event requested by the question.
- If memories conflict, use the latest relevant update for current-state questions.
- Abstain only when the requested information is genuinely missing or the evidence is about a different entity/context.

Rules:
1. Answer directly and specifically.
2. For preference questions, the final answer may be a useful short recommendation, not just a fact span.
3. For fact questions, keep the final answer concise.
4. If evidence is insufficient, say: The information is not available in the recalled memory.

{evidence_text}

Question: {question}

Work through the instructions in <mem_thinking> tags, then give the final answer after "ANSWER:"."""


STRUCTURED_REASONING_ANSWER_PROMPT = """You answer LongMemEval memory questions using only recalled memory evidence.

The user's question date is {question_date}. Interpret relative time expressions against this date.
Use only the recalled memories and the current question. Do not invent unsupported names, numbers, dates, prices, preferences, or events.

First build a compact scratchpad in this exact structure:

<analysis_table>
For every recalled memory that may affect the answer, write one row:
- source_time:
- source_session:
- evidence:
- extracted_fact:
- candidate_value:
- include_or_exclude:
- reason:
</analysis_table>

<task_logic>
If the question asks for a count, list, sum, total, amount, or comparison:
1. List all included candidates.
2. Merge duplicate mentions of the same item/event/value.
3. Exclude assistant-only examples, generic advice, hypothetical options, and plans unless the user states completion or a dated plan has no contradiction.
4. Compute the final count, list, sum, or comparison from the included candidates only.

If the question asks about time, order, elapsed duration, latest/current status, or a relative date:
1. Resolve the relevant date window against the question date.
2. List dated events or values.
3. Sort them by date.
4. Use the latest relevant user-stated value for current status unless the question asks for a historical state.

If the question asks for a preference or recommendation:
1. Extract the user's preferences, anti-preferences, constraints, owned tools/resources, and prior interests.
2. Tailor the final answer to those constraints even if the destination or surface topic is new.
</task_logic>

<final_check>
- Every final answer item must appear in included candidates.
- Do not count duplicate mentions twice.
- Do not count generic assistant advice as a user fact.
- If the requested information is genuinely missing or about a mismatched entity/context, answer exactly: The information is not available in the recalled memory.
</final_check>

{evidence_text}

Question: {question}

Work through the structure, then give the final answer after "ANSWER:"."""


def get_calibrated_longmemeval_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    prompt = get_longmemeval_prompt(task, question, answer, response, abstention=abstention)
    if abstention:
        return prompt
    if task in {"single-session-user", "single-session-assistant", "multi-session"}:
        prompt = prompt.replace(
            "If the response only contains a subset of the information required by the answer, answer no.",
            "If the correct answer is a named entity with an extra location or descriptive qualifier, and the response contains the main named entity unambiguously, treat it as equivalent and answer yes. If the response only contains a subset of required list, count, or multi-part information, answer no.",
        )
    return prompt


def parse_args() -> argparse.Namespace:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/longmemeval/longmemeval_s_dev100_seed20260520.json")
    parser.add_argument("--source-run", default="runs/lme_s_dev100_06b_oldbaseline_rerun_20260521")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--answer-model", default=os.getenv("ANSWER_MODEL") or os.getenv("ANSWERER_MODEL") or os.getenv("LLM_MODEL"))
    parser.add_argument("--answer-base-url", default=os.getenv("ANSWER_BASE_URL") or os.getenv("ANSWERER_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"))
    parser.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--mode", choices=["answer", "judge-existing"], default="answer")
    parser.add_argument(
        "--answer-style",
        choices=[
            "calibrated_short",
            "mem0_lme",
            "global_reasoning",
            "typed_route_mem0_kumstr",
            "structured_reasoning",
            "structured_route_kumstr",
        ],
        default="calibrated_short",
    )
    parser.add_argument("--judge-style", choices=["official", "calibrated"], default="official")
    parser.add_argument("--question-ids")
    parser.add_argument("--question-types")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.judge_model:
        raise RuntimeError("JUDGE_MODEL is required.")
    run_dir = ensure_dir(Path(args.output_dir) / args.run_id)
    source_run = Path(args.source_run)
    samples = load_samples(
        args.data,
        question_ids=load_question_ids(args.question_ids),
        question_types=parse_question_types(args.question_types),
    )
    sample_by_id = {sample.question_id: sample for sample in samples}
    retrieval_by_id = {row["question_id"]: row for row in read_jsonl(source_run / "retrieval_results.jsonl")}
    write_json(run_dir / "config.json", {**vars(args), "num_selected_samples": len(samples)})

    if args.mode == "judge-existing":
        write_predictions(run_dir / "predictions.jsonl", [{"question_id": qid, "hypothesis": value} for qid, value in read_predictions(source_run / "predictions.jsonl").items()])
        if (source_run / "answer_logs.jsonl").exists():
            write_jsonl(run_dir / "answer_logs.jsonl", read_jsonl(source_run / "answer_logs.jsonl"))
    else:
        answer_logs = run_answer_stage(args, run_dir, samples, retrieval_by_id)
        write_token_stats(run_dir, answer_logs, [])

    judge_logs = run_judge_stage(args, run_dir, samples)
    answer_logs = read_jsonl(run_dir / "answer_logs.jsonl")
    write_token_stats(run_dir, answer_logs, judge_logs)
    write_metrics(run_dir, samples, retrieval_by_id, judge_logs)


def run_answer_stage(args: argparse.Namespace, run_dir: Path, samples: list[Any], retrieval_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not args.answer_model:
        raise RuntimeError("ANSWER_MODEL or ANSWERER_MODEL is required.")
    pred_path = run_dir / "predictions.jsonl"
    log_path = run_dir / "answer_logs.jsonl"
    existing_predictions = read_predictions(pred_path) if args.resume and pred_path.exists() else {}
    existing_logs = {
        row["question_id"]: row
        for row in (read_jsonl(log_path) if args.resume and log_path.exists() else [])
        if "question_id" in row
    }
    if not args.resume:
        pred_path.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)

    def answer_one(sample: Any) -> tuple[dict[str, str], dict[str, Any]]:
        retrieval = retrieval_by_id[sample.question_id]
        evidence = format_evidence_for_answerer(
            retrieval["deduped_evidence"],
            sample.question_date,
            question_type=sample.question_type,
        )
        messages = build_answer_messages(args.answer_style, sample.question_date, evidence.text, sample.question, sample.question_type)
        started = time.time()
        body = chat(args.answer_model, args.answer_base_url, os.getenv("ANSWER_API_KEY") or os.getenv("ANSWERER_API_KEY") or os.getenv("OPENAI_API_KEY"), messages, f"answer_prompt_{args.answer_style}")
        latency = time.time() - started
        raw_hypothesis = body["choices"][0].get("message", {}).get("content", "").strip()
        hypothesis = final_answer(args.answer_style, raw_hypothesis)
        usage = dict(body.get("usage", {}))
        prompt_tokens = int(usage.get("prompt_tokens") or estimate_messages_tokens(messages))
        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(raw_hypothesis))
        log = {
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "model": args.answer_model,
            "latency_seconds": latency,
            "query_input_tokens": prompt_tokens,
            "query_output_tokens": completion_tokens,
            "query_total_tokens": prompt_tokens + completion_tokens,
            "answer_prompt_style": args.answer_style,
            "raw_hypothesis": raw_hypothesis,
            "answer_input_tokens": prompt_tokens,
            "answer_output_tokens": completion_tokens,
            "answer_total_tokens": prompt_tokens + completion_tokens,
            "provider_usage": usage,
            "evidence_ids": evidence.included_turn_ids,
            "top_k": retrieval.get("top_k"),
            "message_range": retrieval.get("message_range"),
            "retrieval_method": retrieval.get("retrieval_method"),
            "temporal_boost": retrieval.get("temporal_boost"),
            "evidence_truncated": evidence.truncated,
        }
        return {"question_id": sample.question_id, "hypothesis": hypothesis}, log

    pending_samples = [sample for sample in samples if sample.question_id not in existing_predictions]
    predictions = [{"question_id": question_id, "hypothesis": hypothesis} for question_id, hypothesis in existing_predictions.items()]
    logs = list(existing_logs.values())
    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = [executor.submit(answer_one, sample) for sample in pending_samples]
        for future in as_completed(futures):
            prediction, log = future.result()
            predictions.append(prediction)
            logs.append(log)
            append_jsonl(pred_path, prediction)
            append_jsonl(log_path, log)
    return logs


def build_answer_messages(answer_style: str, question_date: str, evidence_text: str, question: str, question_type: str) -> list[dict[str, str]]:
    if answer_style == "structured_route_kumstr":
        if question_type in {"knowledge-update", "multi-session", "temporal-reasoning"}:
            prompt = STRUCTURED_REASONING_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
            return [{"role": "user", "content": prompt}]
        return build_baseline_answer_messages(question_date, evidence_text, question, question_type)
    if answer_style == "typed_route_mem0_kumstr":
        if question_type in {"knowledge-update", "multi-session", "temporal-reasoning"}:
            prompt = MEM0_LME_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
            return [{"role": "user", "content": prompt}]
        return build_baseline_answer_messages(question_date, evidence_text, question, question_type)
    if answer_style == "structured_reasoning":
        prompt = STRUCTURED_REASONING_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
        return [{"role": "user", "content": prompt}]
    if answer_style == "mem0_lme":
        prompt = MEM0_LME_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
        return [{"role": "user", "content": prompt}]
    if answer_style == "global_reasoning":
        prompt = GLOBAL_REASONING_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
        return [{"role": "user", "content": prompt}]
    system = ANSWER_SYSTEM_PROMPT.format(question_date=question_date)
    instruction = ANSWER_INSTRUCTIONS.get(question_type, "Answer with the minimal span or phrase that directly satisfies the question.")
    user = (
        f"{evidence_text}\n\n"
        "<QUESTION>\n"
        f"{question}\n"
        "</QUESTION>\n\n"
        "<TASK>\n"
        f"{instruction}\n"
        "Return one concise final answer. Do not include citations, evidence IDs, or reasoning.\n"
        "</TASK>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def final_answer(answer_style: str, text: str) -> str:
    if answer_style in {"mem0_lme", "global_reasoning", "typed_route_mem0_kumstr", "structured_reasoning", "structured_route_kumstr"} and "ANSWER:" in text:
        return text.rsplit("ANSWER:", 1)[1].strip()
    return text


def parse_question_types(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def run_judge_stage(args: argparse.Namespace, run_dir: Path, samples: list[Any]) -> list[dict[str, Any]]:
    pred_by_id = read_predictions(run_dir / "predictions.jsonl")
    log_path = run_dir / "judge_logs.jsonl"
    existing = {
        row["question_id"]: row
        for row in (read_jsonl(log_path) if args.resume and log_path.exists() else [])
        if "question_id" in row
    }
    if not args.resume:
        log_path.unlink(missing_ok=True)

    def judge_one(sample: Any) -> dict[str, Any]:
        response = pred_by_id[sample.question_id]
        if args.judge_style == "official":
            prompt = get_longmemeval_prompt(sample.question_type, sample.question, sample.answer, response, abstention=is_abstention_sample(sample))
        else:
            prompt = get_calibrated_longmemeval_prompt(sample.question_type, sample.question, sample.answer, response, abstention=is_abstention_sample(sample))
        messages = [{"role": "user", "content": prompt}]
        started = time.time()
        body = chat(args.judge_model, args.judge_base_url, os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY"), messages, f"judge_{args.judge_style}", max_tokens=10)
        latency = time.time() - started
        content = body["choices"][0].get("message", {}).get("content", "").strip()
        usage = dict(body.get("usage", {}))
        prompt_tokens = int(usage.get("prompt_tokens") or estimate_messages_tokens(messages))
        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(content))
        label = "CORRECT" if "yes" in content.lower() else "WRONG"
        row = {
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "model": args.judge_model,
            "judge_style": args.judge_style,
            "label": label,
            "score": 1.0 if label == "CORRECT" else 0.0,
            "raw_response": content,
            "judge_input_tokens": prompt_tokens,
            "judge_output_tokens": completion_tokens,
            "judge_total_tokens": prompt_tokens + completion_tokens,
            "provider_usage": usage,
            "latency_seconds": latency,
        }
        return row

    logs_by_id = dict(existing)
    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = [
            executor.submit(judge_one, sample)
            for sample in samples
            if sample.question_id in pred_by_id and sample.question_id not in existing
        ]
        for future in as_completed(futures):
            row = future.result()
            logs_by_id[row["question_id"]] = row
            append_jsonl(log_path, row)
    return [logs_by_id[sample.question_id] for sample in samples if sample.question_id in logs_by_id]


def chat(model: str, base_url: str | None, api_key: str | None, messages: list[dict[str, str]], cache_kind: str, max_tokens: int | None = None) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY-compatible key is required.")
    payload_obj: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
    if max_tokens is not None:
        payload_obj["max_tokens"] = max_tokens
    cached = cached_response(cache_kind, model, payload_obj)
    if cached is not None:
        return cached
    payload = json.dumps(payload_obj).encode("utf-8")
    request = urllib.request.Request(
        f"{(base_url or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1').rstrip('/')}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except (urllib.error.URLError, TimeoutError, socket.timeout, http.client.RemoteDisconnected) as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"chat request failed: {last_error}")
    write_cached_response(cache_kind, model, payload_obj, body)
    return body


def write_token_stats(run_dir: Path, answer_logs: list[dict[str, Any]], judge_logs: list[dict[str, Any]]) -> None:
    stats = new_token_summary()
    for row in answer_logs:
        add_query_tokens(stats, row["question_id"], row.get("query_input_tokens", 0), row.get("query_output_tokens", 0))
        add_query_time(stats, row["question_id"], row.get("latency_seconds", 0.0))
    for row in judge_logs:
        add_judge_tokens(stats, row["question_id"], row.get("judge_input_tokens", 0), row.get("judge_output_tokens", 0))
        add_judge_time(stats, row["question_id"], row.get("latency_seconds", 0.0))
    write_json(run_dir / "token_stats.json", stats)


def write_metrics(run_dir: Path, samples: list[Any], retrieval_by_id: dict[str, dict[str, Any]], judge_logs: list[dict[str, Any]]) -> None:
    metrics = qa_metrics(judge_logs)
    write_json(run_dir / "qa_metrics.json", metrics)
    write_json(run_dir / "metrics.json", {"qa": metrics})
    if retrieval_by_id:
        write_jsonl(run_dir / "retrieval_results.jsonl", [retrieval_by_id[sample.question_id] for sample in samples if sample.question_id in retrieval_by_id])
    write_error_analysis(
        samples,
        read_jsonl(run_dir / "retrieval_results.jsonl"),
        run_dir / "predictions.jsonl",
        json.loads((run_dir / "token_stats.json").read_text(encoding="utf-8")),
        run_dir / "error_analysis.jsonl",
        autoeval_log=run_dir / "judge_logs.jsonl",
    )
    print(f"accuracy={metrics['overall_accuracy']:.4f} correct={metrics['correct']} total={metrics['total']} run_dir={run_dir}")


def qa_metrics(logs: list[dict[str, Any]]) -> dict[str, Any]:
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


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
