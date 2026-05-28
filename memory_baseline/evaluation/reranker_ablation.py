from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_baseline.core.utils import ensure_dir, estimate_tokens, read_jsonl, timestamp_sort_key, write_json, write_jsonl
from memory_baseline.retrieval.dense import _dedupe_and_sort_windows


DEFAULT_RERANKER = "infgrad/Prism-Qwen3.5-Reranker-0.8B"
DEFAULT_ETTIN_RERANKER = "cross-encoder/ettin-reranker-150m-v1"

SYSTEM_PROMPT = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
INSTRUCTION = (
    'Judge if the document is relevant to the query. Reply "yes" or "no".\n'
    'On "yes", also emit:\n'
    "<contribution>One sentence covering every core point the document contributes to the query, without elaboration.</contribution>\n"
    "<evidence>Self-contained rewrite of the query-relevant content. Rules:\n"
    "- Faithful: rephrase only; add or infer nothing.\n"
    "- Self-contained: evidence alone must fully answer the query.\n"
    "- Concise: drop query-irrelevant background.\n"
    "- Verbatim (no translation): proper nouns, terms, abbreviations, numbers, dates, code, URLs.\n"
    "- Output language: multilingual doc -> query's language; else doc's language."
    "</evidence>"
)
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "<Instruct>: {instruction}\n"
    "<Query>: {query}\n"
    "<Document>: {doc}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def main() -> None:
    args = parse_args()
    if args.model is None:
        args.model = DEFAULT_ETTIN_RERANKER if args.reranker_kind == "cross-encoder" else DEFAULT_RERANKER
    run_dir = Path(args.retrieval_run)
    out_dir = ensure_dir(Path(args.output_dir))
    reranker = load_reranker(args)
    rows = []
    reranked_results = []
    for row_idx, result in enumerate(read_jsonl(run_dir / "retrieval_results.jsonl")):
        if row_idx % args.num_shards != args.shard_index:
            continue
        if args.question_types and result.get("question_type") not in args.question_types:
            continue
        if args.candidate_unit == "window":
            candidates = result.get("evidence_windows", [])
            source_turns = result.get("deduped_evidence", [])
        else:
            candidates = result.get("deduped_evidence", [])
            source_turns = candidates
        scored = score_candidates(reranker, result["question"], candidates, args.batch_size, args.max_length)
        selected = scored[: args.keep_turns]
        if args.candidate_unit == "window":
            selected_windows = sorted(
                [item["turn"] for item in selected],
                key=lambda window: (
                    timestamp_sort_key((window.get("turns") or [{}])[0].get("session_date")),
                    int((window.get("turns") or [{}])[0].get("turn_idx", 0)),
                ),
            )
            selected_turns = _dedupe_and_sort_windows(selected_windows)
        else:
            selected = sorted(
                selected,
                key=lambda item: (
                    timestamp_sort_key(item["turn"].get("session_date")),
                    int(item["turn"].get("turn_idx", 0)),
                ),
            )
            selected_windows = []
            selected_turns = [item["turn"] for item in selected]
        reranked_result = dict(result)
        reranked_result["deduped_evidence"] = selected_turns
        if args.candidate_unit == "window":
            reranked_result["evidence_windows"] = selected_windows
        reranked_result["reranker"] = {
            "kind": args.reranker_kind,
            "model": args.model,
            "candidate_unit": args.candidate_unit,
            "keep_turns": args.keep_turns,
        }
        reranked_results.append(reranked_result)
        row = {
            "question_id": result["question_id"],
            "question_type": result.get("question_type"),
            "question": result.get("question"),
            "answer": result.get("answer"),
            "keep_turns": args.keep_turns,
            "source_turns": len(source_turns),
            "selected_turns": len(selected),
            "selected_evidence_turns": len(selected_turns),
            "source_tokens": estimate_tokens("\n".join(str(turn.get("content", "")) for turn in source_turns)),
            "selected_tokens": estimate_tokens("\n".join(str(turn.get("content", "")) for turn in selected_turns)),
            "source_sessions": len({str(turn.get("session_id")) for turn in source_turns}),
            "selected_sessions": len({str(turn.get("session_id")) for turn in selected_turns}),
            "source_turn_hit": any(turn.get("has_answer") for turn in source_turns),
            "selected_turn_hit": any(turn.get("has_answer") for turn in selected_turns),
            "source_session_hit": session_hit(result.get("answer_session_ids", []), source_turns),
            "selected_session_hit": session_hit(result.get("answer_session_ids", []), selected_turns),
            "source_all_session_hit": all_session_hit(result.get("answer_session_ids", []), source_turns),
            "selected_all_session_hit": all_session_hit(result.get("answer_session_ids", []), selected_turns),
            "selected": [
                {
                    "rerank_score": item["score"],
                    **selected_item_preview(item["turn"], args.candidate_unit),
                }
                for item in selected
            ],
        }
        rows.append(row)
    summary = summarize(rows)
    write_json(out_dir / "reranker_ablation_summary.json", summary)
    write_jsonl(out_dir / "reranker_ablation_rows.jsonl", rows)
    write_jsonl(out_dir / "retrieval_results.jsonl", reranked_results)
    (out_dir / "reranker_ablation.md").write_text(render_markdown(summary, rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-run", default="runs/lme_dev100_ms_wrong_top100_retrieval_20260523")
    parser.add_argument("--output-dir", default="runs/_diagnostics/reranker_ablation_ms_wrong_top100_20260523")
    parser.add_argument("--reranker-kind", choices=["prism", "cross-encoder"], default="prism")
    parser.add_argument("--model")
    parser.add_argument("--keep-turns", type=int, default=20)
    parser.add_argument("--candidate-unit", choices=["turn", "window"], default="turn")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--question-types", nargs="*", default=["multi-session"])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def load_reranker(args: argparse.Namespace) -> dict[str, Any]:
    if args.reranker_kind == "cross-encoder":
        return load_cross_encoder_reranker(args.model, args.max_length, args.attn_implementation)
    tokenizer, model, device = load_prism_reranker(args.model)
    return {"kind": "prism", "tokenizer": tokenizer, "model": model, "device": device}


def load_cross_encoder_reranker(model_path: str, max_length: int, attn_implementation: str) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(
        model_path,
        device=device,
        trust_remote_code=True,
        max_length=max_length,
        model_kwargs={"dtype": torch.bfloat16, "attn_implementation": attn_implementation},
    )
    return {"kind": "cross-encoder", "model": model}


def load_prism_reranker(model_path: str) -> tuple[Any, Any, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def score_candidates(reranker: dict[str, Any], query: str, candidates: list[dict[str, Any]], batch_size: int, max_length: int) -> list[dict[str, Any]]:
    if reranker["kind"] == "cross-encoder":
        return score_cross_encoder_candidates(reranker["model"], query, candidates, batch_size)
    return score_prism_candidates(reranker["tokenizer"], reranker["model"], reranker["device"], query, candidates, batch_size, max_length)


def score_cross_encoder_candidates(model: CrossEncoder, query: str, candidates: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    pairs = [(query, document_text(candidate)) for candidate in candidates]
    scores = model.predict(pairs, batch_size=batch_size)
    scored = [{"score": float(score), "turn": candidate} for score, candidate in zip(scores, candidates)]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def score_prism_candidates(tokenizer: Any, model: Any, device: torch.device, query: str, candidates: list[dict[str, Any]], batch_size: int, max_length: int) -> list[dict[str, Any]]:
    scored = []
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    prompts = [build_prompt(tokenizer, query, candidate) for candidate in candidates]
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            logits = generated.scores[0][:, [no_id, yes_id]].float()
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
        for offset, score in enumerate(probs):
            scored.append({"score": float(score), "turn": candidates[start + offset]})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def build_prompt(tokenizer: Any, query: str, candidate: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, instruction=INSTRUCTION, query=query, doc=document_text(candidate))


def document_text(candidate: dict[str, Any]) -> str:
    if "turns" in candidate:
        turns = candidate.get("turns") or []
        lines = [f"Matched turn: {candidate.get('matched_stable_turn_id', '')}"]
        for turn in turns:
            lines.append(
                f"Date: {turn.get('session_date', '')}\n"
                f"Session: {turn.get('session_id', '')}\n"
                f"Role: {turn.get('role', '')}\n"
                f"Turn: {turn.get('turn_idx', '')}\n"
                f"Document: {turn.get('content', '')}"
            )
        return "\n\n".join(lines)
    return (
        f"Date: {candidate.get('session_date', '')}\n"
        f"Role: {candidate.get('role', '')}\n"
        f"Document: {candidate.get('content', '')}"
    )


def selected_item_preview(candidate: dict[str, Any], candidate_unit: str) -> dict[str, Any]:
    if candidate_unit == "window":
        turns = candidate.get("turns") or []
        return {
            "matched_stable_turn_id": candidate.get("matched_stable_turn_id"),
            "start_turn_idx": candidate.get("start_turn_idx"),
            "end_turn_idx": candidate.get("end_turn_idx"),
            "session_id": turns[0].get("session_id") if turns else None,
            "has_answer": any(turn.get("has_answer", False) for turn in turns),
            "content": "\n".join(str(turn.get("content", "")) for turn in turns),
        }
    return {
        "stable_turn_id": candidate.get("stable_turn_id"),
        "session_id": candidate.get("session_id"),
        "turn_idx": candidate.get("turn_idx"),
        "has_answer": candidate.get("has_answer", False),
        "content": candidate.get("content", ""),
    }


def session_hit(answer_session_ids: list[Any], turns: list[dict[str, Any]]) -> bool:
    answer_sessions = {str(session_id) for session_id in answer_session_ids}
    return bool(answer_sessions & {str(turn.get("session_id")) for turn in turns})


def all_session_hit(answer_session_ids: list[Any], turns: list[dict[str, Any]]) -> bool:
    answer_sessions = {str(session_id) for session_id in answer_session_ids}
    recalled = {str(turn.get("session_id")) for turn in turns}
    return answer_sessions <= recalled


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "total": total,
        "source_turn_recall": ratio(row["source_turn_hit"] for row in rows),
        "selected_turn_recall": ratio(row["selected_turn_hit"] for row in rows),
        "source_session_recall": ratio(row["source_session_hit"] for row in rows),
        "selected_session_recall": ratio(row["selected_session_hit"] for row in rows),
        "source_all_session_recall": ratio(row["source_all_session_hit"] for row in rows),
        "selected_all_session_recall": ratio(row["selected_all_session_hit"] for row in rows),
        "avg_source_tokens": mean(row["source_tokens"] for row in rows),
        "avg_selected_tokens": mean(row["selected_tokens"] for row in rows),
        "avg_source_sessions": mean(row["source_sessions"] for row in rows),
        "avg_selected_sessions": mean(row["selected_sessions"] for row in rows),
    }


def render_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Reranker Ablation",
        "",
        f"Samples: `{summary['total']}`",
        f"Turn recall: `{summary['source_turn_recall']:.3f}` -> `{summary['selected_turn_recall']:.3f}`",
        f"Session recall: `{summary['source_session_recall']:.3f}` -> `{summary['selected_session_recall']:.3f}`",
        f"All answer-session recall: `{summary['source_all_session_recall']:.3f}` -> `{summary['selected_all_session_recall']:.3f}`",
        f"Avg tokens: `{summary['avg_source_tokens']:.1f}` -> `{summary['avg_selected_tokens']:.1f}`",
        f"Avg sessions: `{summary['avg_source_sessions']:.1f}` -> `{summary['avg_selected_sessions']:.1f}`",
        "",
        "| QID | Turn hit | Any session hit | All sessions hit | Tokens | Sessions | Question |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['question_id']}` | {row['source_turn_hit']} -> {row['selected_turn_hit']} | "
            f"{row['source_session_hit']} -> {row['selected_session_hit']} | "
            f"{row['source_all_session_hit']} -> {row['selected_all_session_hit']} | "
            f"{row['source_tokens']} -> {row['selected_tokens']} | "
            f"{row['source_sessions']} -> {row['selected_sessions']} | {row['question']} |"
        )
    lines.append("")
    return "\n".join(lines)


def ratio(values: Any) -> float:
    items = [bool(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def mean(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


if __name__ == "__main__":
    main()
