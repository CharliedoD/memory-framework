from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np

from memory_baseline.core.env import load_project_env
from memory_baseline.core.io import write_predictions
from memory_baseline.core.token_accounting import (
    add_build_time,
    add_build_tokens,
    add_judge_time,
    add_judge_tokens,
    add_query_time,
    add_query_tokens,
    add_retrieval_embedding_tokens,
    new_token_summary,
)
from memory_baseline.core.utils import ensure_dir, estimate_tokens, read_jsonl, safe_filename, write_json, write_jsonl
from memory_baseline.data.locomo import (
    CATEGORY_NAMES,
    conversation_raw_turns,
    iter_qa_items,
    load_locomo_records,
    parse_int_list,
)
from memory_baseline.generation.answerer import make_answerer
from memory_baseline.generation.judge import make_judge
from memory_baseline.generation.provence_pruner import (
    DEFAULT_PROVENCE_MODEL,
    ProvenceEvidencePruner,
    ProvencePruningResult,
    make_provence_pruner,
)
from memory_baseline.indexing.embedder import embed_texts_cached, make_embedder, release_cuda_cache
from memory_baseline.indexing.vector_store import QuestionStore, embedding_texts_for_turns
from memory_baseline.retrieval.dense import _dedupe_and_sort_windows, _expand_windows, _filter_indices, _matched_turn, compute_retrieval_metrics
from memory_baseline.retrieval.formatter import format_evidence_for_answerer
from memory_baseline.retrieval.ranking import rank_indices
from memory_baseline.retrieval.typed_facts import build_typed_facts, prepend_typed_fact_pack, select_typed_facts, should_use_typed_facts, source_turn_ids


LOCOMO_OVERALL_CATEGORIES = {1, 2, 3, 4}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/locomo/locomo10.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--categories", default="1,2,3,4")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-k-list")
    parser.add_argument("--message-range", type=int, default=2)
    parser.add_argument("--chunk-mode", choices=["turn"], default="turn")
    parser.add_argument("--retrieval-method", choices=["dense", "hybrid", "semantic_bm25_boost"], default="dense")
    parser.add_argument("--temporal-boost", type=float, default=0.0)
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL") or os.getenv("EMBEDDER_MODEL") or "local-hash")
    parser.add_argument("--embedding-base-url", default=os.getenv("EMBEDDING_BASE_URL") or os.getenv("EMBEDDER_BASE_URL"))
    parser.add_argument("--embedding-backend", choices=["auto", "api", "hf", "sentence-transformers"], default=os.getenv("EMBEDDING_BACKEND", "auto"))
    parser.add_argument("--answer-model", default=os.getenv("ANSWER_MODEL") or os.getenv("ANSWERER_MODEL") or os.getenv("LLM_MODEL"))
    parser.add_argument("--answer-base-url", default=os.getenv("ANSWER_BASE_URL") or os.getenv("ANSWERER_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"))
    parser.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--answer-parallelism", type=int)
    parser.add_argument("--judge-parallelism", type=int)
    parser.add_argument("--cutoff-parallelism", type=int, default=1)
    parser.add_argument("--mode", choices=["build", "retrieve", "answer", "judge", "eval", "full", "index", "api"], default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-evidence-tokens", type=int)
    parser.add_argument("--provence-pruning", action="store_true")
    parser.add_argument("--provence-model", default=DEFAULT_PROVENCE_MODEL)
    parser.add_argument("--provence-threshold", type=float, default=0.1)
    parser.add_argument("--provence-batch-size", type=int, default=32)
    parser.add_argument("--typed-sidecar", action="store_true")
    parser.add_argument("--output-dir", default="runs")
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "full" and os.getenv("MEMORY_BASELINE_LOCOMO_SPLIT_CHILD") != "1":
        return run_full_split(args)

    cutoffs = _top_k_cutoffs(args)
    if len(cutoffs) > 1:
        return run_topk_pipeline(args, cutoffs)

    run_dir = ensure_dir(Path(args.output_dir) / args.run_id)
    records = load_locomo_records(args.data)
    conversations = parse_int_list(args.conversations, list(range(len(records))))
    categories = parse_int_list(args.categories, [1, 2, 3, 4])
    qa_items = iter_qa_items(records, conversations, categories, question_limit=args.question_limit)
    write_json(run_dir / "config.json", _config_dict(args, conversations, categories, qa_items))
    token_stats = _load_or_new_token_stats(run_dir)

    if args.mode in {"build", "full", "index"}:
        build_stats = run_build_stage(args, run_dir, records, conversations)
        _reset_build(token_stats)
        for stat in build_stats:
            add_build_tokens(token_stats, stat["conversation_id"], stat.get("build_embedding_input_tokens", 0), stat.get("build_embedding_provider_tokens", 0))
            add_build_time(token_stats, stat["conversation_id"], stat.get("build_time_seconds", 0.0))
        write_json(run_dir / "token_stats.json", token_stats)

    if args.mode in {"retrieve", "full", "index"}:
        retrieval_results = run_retrieve_stage(args, run_dir, records, qa_items)
        _reset_retrieval(token_stats)
        for result in retrieval_results:
            add_retrieval_embedding_tokens(token_stats, result["question_id"], result.get("query_embedding_tokens", 0))
        write_json(run_dir / "token_stats.json", token_stats)

    if args.mode in {"answer", "full", "api"}:
        answer_logs = run_answer_stage(args, run_dir, qa_items)
        _reset_query(token_stats)
        retrieval_by_id = {row["question_id"]: row for row in read_jsonl(run_dir / "retrieval_results.jsonl")}
        for log in answer_logs:
            retrieval_latency = retrieval_by_id.get(log["question_id"], {}).get("retrieval_latency_seconds", 0.0)
            query_time = float(retrieval_latency) + float(log.get("latency_seconds", 0.0))
            add_query_tokens(token_stats, log["question_id"], log.get("query_input_tokens", 0), log.get("query_output_tokens", 0))
            add_query_time(token_stats, log["question_id"], query_time)
        write_json(run_dir / "token_stats.json", token_stats)

    if args.mode in {"judge", "full", "api"}:
        judge_logs = run_judge_stage(args, run_dir, qa_items)
        _reset_judge(token_stats)
        for log in judge_logs:
            add_judge_tokens(token_stats, log["question_id"], log.get("judge_input_tokens", 0), log.get("judge_output_tokens", 0))
            add_judge_time(token_stats, log["question_id"], log.get("latency_seconds", 0.0))
        write_json(run_dir / "token_stats.json", token_stats)

    if args.mode in {"eval", "full", "api"}:
        write_metrics(run_dir, categories)

    return {"run_dir": str(run_dir), "num_questions": len(qa_items), "num_conversations": len(conversations)}


def run_full_split(args: argparse.Namespace) -> dict[str, Any]:
    records = load_locomo_records(args.data)
    conversations = parse_int_list(args.conversations, list(range(len(records))))
    categories = parse_int_list(args.categories, [1, 2, 3, 4])
    qa_items = iter_qa_items(records, conversations, categories, question_limit=args.question_limit)
    _run_stage_subprocess(args, "index", keep_cuda=True)
    _run_stage_subprocess(args, "api", keep_cuda=False)
    return {"run_dir": str(Path(args.output_dir) / args.run_id), "num_questions": len(qa_items), "num_conversations": len(conversations)}


def run_topk_pipeline(args: argparse.Namespace, cutoffs: list[int]) -> dict[str, Any]:
    run_dir = ensure_dir(Path(args.output_dir) / args.run_id)
    records = load_locomo_records(args.data)
    conversations = parse_int_list(args.conversations, list(range(len(records))))
    categories = parse_int_list(args.categories, [1, 2, 3, 4])
    qa_items = iter_qa_items(records, conversations, categories, question_limit=args.question_limit)
    max_top_k = max(cutoffs)
    max_args = argparse.Namespace(**vars(args))
    max_args.top_k = max_top_k
    config = _config_dict(max_args, conversations, categories, qa_items)
    config["top_k_list"] = cutoffs
    write_json(run_dir / "config.json", config)
    token_stats = _load_or_new_token_stats(run_dir)

    if args.mode in {"build", "full", "index"}:
        build_stats = run_build_stage(max_args, run_dir, records, conversations)
        _reset_build(token_stats)
        for stat in build_stats:
            add_build_tokens(token_stats, stat["conversation_id"], stat.get("build_embedding_input_tokens", 0), stat.get("build_embedding_provider_tokens", 0))
            add_build_time(token_stats, stat["conversation_id"], stat.get("build_time_seconds", 0.0))
        write_json(run_dir / "token_stats.json", token_stats)

    if args.mode in {"retrieve", "full", "index"}:
        retrieval_results = run_retrieve_stage(max_args, run_dir, records, qa_items)
        _reset_retrieval(token_stats)
        for result in retrieval_results:
            add_retrieval_embedding_tokens(token_stats, result["question_id"], result.get("query_embedding_tokens", 0))
        write_json(run_dir / "token_stats.json", token_stats)

    if args.mode in {"answer", "judge", "eval", "full", "api"}:
        def run_cutoff(cutoff: int) -> int:
            cutoff_dir = ensure_dir(run_dir / f"top_{cutoff}")
            _prepare_cutoff_run_dir(run_dir, cutoff_dir, cutoff, args.max_evidence_tokens)
            cutoff_args = argparse.Namespace(**vars(args))
            cutoff_args.top_k = cutoff
            cutoff_token_stats = _load_cutoff_token_stats(run_dir, cutoff_dir, args)
            if args.mode in {"answer", "full", "api"}:
                answer_logs = run_answer_stage(cutoff_args, cutoff_dir, qa_items)
                _reset_query(cutoff_token_stats)
                retrieval_by_id = {row["question_id"]: row for row in read_jsonl(cutoff_dir / "retrieval_results.jsonl")}
                for log in answer_logs:
                    retrieval_latency = retrieval_by_id.get(log["question_id"], {}).get("retrieval_latency_seconds", 0.0)
                    query_time = float(retrieval_latency) + float(log.get("latency_seconds", 0.0))
                    add_query_tokens(cutoff_token_stats, log["question_id"], log.get("query_input_tokens", 0), log.get("query_output_tokens", 0))
                    add_query_time(cutoff_token_stats, log["question_id"], query_time)
                write_json(cutoff_dir / "token_stats.json", cutoff_token_stats)
            if args.mode in {"judge", "full", "api"}:
                judge_logs = run_judge_stage(cutoff_args, cutoff_dir, qa_items)
                _reset_judge(cutoff_token_stats)
                for log in judge_logs:
                    add_judge_tokens(cutoff_token_stats, log["question_id"], log.get("judge_input_tokens", 0), log.get("judge_output_tokens", 0))
                    add_judge_time(cutoff_token_stats, log["question_id"], log.get("latency_seconds", 0.0))
                write_json(cutoff_dir / "token_stats.json", cutoff_token_stats)
            if args.mode in {"eval", "full", "api"}:
                write_metrics(cutoff_dir, categories)
            return cutoff

        for _cutoff in _iter_parallel(cutoffs, run_cutoff, args.cutoff_parallelism):
            pass
        if args.mode in {"eval", "full", "api"}:
            write_topk_summary(run_dir, cutoffs)

    return {"run_dir": str(run_dir), "num_questions": len(qa_items), "num_conversations": len(conversations), "top_k_list": cutoffs}


def run_build_stage(args: argparse.Namespace, run_dir: Path, records: list[dict[str, Any]], conversations: list[int]) -> list[dict[str, Any]]:
    embedder = make_embedder(args.embedding_model, args.embedding_base_url, backend=_backend_arg(args.embedding_backend))
    try:
        stats = []
        for conv_idx in conversations:
            stats.append(
                build_conversation_store(
                    records[conv_idx],
                    conv_idx,
                    run_dir,
                    embedder,
                    args.force_rebuild,
                    args.skip_existing or args.resume,
                    args.chunk_mode,
                    args.typed_sidecar,
                )
            )
        write_jsonl(run_dir / "build_stats.jsonl", stats)
        return stats
    finally:
        del embedder
        release_cuda_cache()


def build_conversation_store(
    record: dict[str, Any],
    conv_idx: int,
    run_dir: Path,
    embedder: Any,
    force_rebuild: bool,
    skip_existing: bool,
    chunk_mode: str,
    typed_sidecar: bool = False,
) -> dict[str, Any]:
    store_dir = _store_dir(run_dir, conv_idx)
    stats_path = store_dir / "store_stats.json"
    if (store_dir / "embeddings.npy").exists() and not force_rebuild and stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        _check_chunk_mode(stats, chunk_mode, store_dir)
        if typed_sidecar and not (store_dir / "typed_facts.jsonl").exists():
            typed_facts = build_typed_facts(read_jsonl(store_dir / "raw_turns.jsonl"))
            write_jsonl(store_dir / "typed_facts.jsonl", typed_facts)
            stats["typed_sidecar"] = True
            stats["typed_fact_count"] = len(typed_facts)
            write_json(stats_path, stats)
        return stats
    if store_dir.exists() and force_rebuild:
        shutil.rmtree(store_dir)
    ensure_dir(store_dir)

    started = time.monotonic()
    raw_turns = conversation_raw_turns(record, conv_idx)
    texts = embedding_texts_for_turns(raw_turns, chunk_mode)
    batch = embed_texts_cached(embedder, texts, cache_root=Path(".cache") / "embeddings", force=force_rebuild)
    np.save(store_dir / "embeddings.npy", batch.vectors)
    write_jsonl(store_dir / "raw_turns.jsonl", raw_turns)
    write_jsonl(store_dir / "embedding_meta.jsonl", ({**turn, "embedding_text": text} for turn, text in zip(raw_turns, texts)))
    typed_fact_count = 0
    if typed_sidecar:
        typed_facts = build_typed_facts(raw_turns)
        write_jsonl(store_dir / "typed_facts.jsonl", typed_facts)
        typed_fact_count = len(typed_facts)
    stats = {
        "conversation_id": f"conv{conv_idx}",
        "conversation_idx": conv_idx,
        "num_turns": len(raw_turns),
        "chunk_mode": chunk_mode,
        "typed_sidecar": typed_sidecar,
        "typed_fact_count": typed_fact_count,
        "embedder_model": embedder.model_name,
        "build_embedding_input_tokens": batch.input_tokens,
        "build_embedding_provider_tokens": batch.provider_tokens,
        "embedding_cache_hits": batch.cache_hits,
        "embedding_cache_misses": batch.cache_misses,
        "build_time_seconds": time.monotonic() - started,
    }
    write_json(stats_path, stats)
    return stats


def run_retrieve_stage(
    args: argparse.Namespace,
    run_dir: Path,
    records: list[dict[str, Any]],
    qa_items: list[tuple[int, int, dict[str, Any], Any]],
) -> list[dict[str, Any]]:
    embedder = make_embedder(args.embedding_model, args.embedding_base_url, backend=_backend_arg(args.embedding_backend))
    stores: dict[int, QuestionStore] = {}
    try:
        results = []
        for conv_idx, qa_idx, qa, sample in qa_items:
            store = stores.setdefault(conv_idx, QuestionStore(_store_dir(run_dir, conv_idx)))
            _check_store(_store_dir(run_dir, conv_idx), embedder.model_name, args.chunk_mode)
            results.append(
                retrieve_one(
                    store,
                    embedder,
                    sample,
                    qa,
                    args.top_k,
                    args.message_range,
                    args.max_evidence_tokens,
                    args.retrieval_method,
                    args.temporal_boost,
                    args.typed_sidecar,
                )
            )
        write_jsonl(run_dir / "retrieval_results.jsonl", results)
        return results
    finally:
        stores.clear()
        del embedder
        release_cuda_cache()


def retrieve_one(
    store: QuestionStore,
    embedder: Any,
    sample: Any,
    qa: dict[str, Any],
    top_k: int,
    message_range: int,
    max_evidence_tokens: int | None,
    retrieval_method: str = "dense",
    temporal_boost: float = 0.0,
    typed_sidecar: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    evidence_ids = {str(evidence) for evidence in qa.get("evidence", [])}
    answer_sessions = set(sample.answer_session_ids)
    query_batch = embed_texts_cached(embedder, [sample.question], cache_root=Path(".cache") / "embeddings")
    query = query_batch.vectors[0]
    scores = store.embeddings @ query
    candidate_indices = _filter_indices(store.raw_turns, None, sample.question_date)
    ranking_diagnostics: dict[str, Any] = {}
    indices, ranked_scores, _bm25_scores = rank_indices(
        query=sample.question,
        dense_scores=scores,
        raw_turns=store.raw_turns,
        retrieval_texts=store.lexical_texts,
        candidate_indices=candidate_indices,
        top_k=top_k,
        retrieval_method=retrieval_method,
        question_type=sample.question_type,
        question_date=sample.question_date,
        temporal_boost=temporal_boost,
        store=store,
        ranking_diagnostics=ranking_diagnostics,
    )
    matched_turns = []
    for rank, idx in enumerate(indices, start=1):
        turn = _with_dynamic_labels(dict(store.raw_turns[int(idx)]), evidence_ids, answer_sessions)
        matched_turns.append(_matched_turn(turn, float(ranked_scores[int(idx)]), rank))
    evidence_windows = _expand_windows(store.raw_turns, matched_turns, message_range)
    for window in evidence_windows:
        window["turns"] = [_with_dynamic_labels(dict(turn), evidence_ids, answer_sessions) for turn in window["turns"]]
    deduped_evidence = [_with_dynamic_labels(turn, evidence_ids, answer_sessions) for turn in _dedupe_and_sort_windows(evidence_windows)]
    typed_facts = []
    if typed_sidecar and should_use_typed_facts(sample.question, sample.question_type):
        recalled_ids = {str(turn.get("stable_turn_id")) for turn in deduped_evidence}
        candidate_facts = [
            fact
            for fact in store.typed_facts
            if recalled_ids & {str(source_id) for source_id in fact.get("source_turn_ids", [])}
        ]
        typed_facts = select_typed_facts(sample.question, candidate_facts)
    formatted = format_evidence_for_answerer(deduped_evidence, sample.question_date, max_evidence_tokens)
    formatted_text = prepend_typed_fact_pack(formatted.text, typed_facts)
    metrics = compute_retrieval_metrics(sample, matched_turns, deduped_evidence, estimate_tokens(formatted_text))
    result = {
        "question_id": sample.question_id,
        "conversation_idx": int(sample.question_id.split("_q", 1)[0].replace("conv", "")),
        "question_type": sample.question_type,
        "question": sample.question,
        "answer": sample.answer,
        "category": int(qa.get("category", 0)),
        "question_date": sample.question_date,
        "answer_session_ids": sample.answer_session_ids,
        "evidence": list(qa.get("evidence", [])),
        "top_k": top_k,
        "message_range": message_range,
        "retrieval_method": retrieval_method,
        "temporal_boost": temporal_boost,
        "timestamp_filter": {
            "question_date": sample.question_date,
            "num_candidates_before_filter": len(store.raw_turns),
            "num_candidates_after_filter": len(candidate_indices),
        },
        "matched_turns": matched_turns,
        "evidence_windows": evidence_windows,
        "deduped_evidence": deduped_evidence,
        "formatted_evidence": formatted.text,
        "typed_augmented_evidence": formatted_text,
        "typed_sidecar": typed_sidecar,
        "typed_facts": typed_facts,
        "typed_fact_source_turn_ids": sorted(source_turn_ids(typed_facts)),
        "formatted_evidence_turn_ids": formatted.included_turn_ids,
        "evidence_truncated": formatted.truncated,
        "evidence_truncate_strategy": formatted.truncate_strategy,
        "query_embedding_tokens": query_batch.input_tokens,
        "retrieval_latency_seconds": time.monotonic() - started,
        "metrics": metrics,
    }
    if ranking_diagnostics:
        result["ranking_diagnostics"] = ranking_diagnostics
    return result


def run_answer_stage(args: argparse.Namespace, run_dir: Path, qa_items: list[tuple[int, int, dict[str, Any], Any]]) -> list[dict[str, Any]]:
    retrieval_by_id = {row["question_id"]: row for row in read_jsonl(run_dir / "retrieval_results.jsonl")}
    answerer = make_answerer(args.answer_model, args.answer_base_url)
    workers = args.answer_parallelism if args.answer_parallelism is not None else args.parallelism
    pruners = _make_provence_pruners(args, max(1, workers)) if args.provence_pruning else None
    pruner_queue = Queue()
    for pruner in pruners or []:
        pruner_queue.put(pruner)
    skip_existing = args.skip_existing or args.resume
    existing = _existing_by_id(run_dir / "answer_logs.jsonl") if skip_existing else {}
    pred_path = run_dir / "predictions.jsonl"
    log_path = run_dir / "answer_logs.jsonl"
    if not skip_existing:
        pred_path.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)

    def answer_one(item: tuple[int, int, dict[str, Any], Any]) -> tuple[dict[str, str], dict[str, Any]]:
        _conv_idx, _qa_idx, _qa, sample = item
        if sample.question_id in existing:
            log = dict(existing[sample.question_id])
            return {"question_id": sample.question_id, "hypothesis": str(log.get("hypothesis", ""))}, log
        result = retrieval_by_id[sample.question_id]
        pruner = pruner_queue.get() if pruner_queue.qsize() else None
        try:
            formatted = _formatted_evidence_for_answer(
                result,
                sample,
                args.max_evidence_tokens,
                pruner,
            )
        finally:
            if pruner is not None:
                pruner_queue.put(pruner)
        pruning = formatted["pruning"]
        try:
            answer = answerer.answer(sample.question_date, formatted["text"], sample.question)
        except Exception as exc:
            query_input_tokens = estimate_tokens(formatted["text"]) + estimate_tokens(sample.question)
            log = {
                "question_id": sample.question_id,
                "model": answerer.model_name,
                "hypothesis": "ERROR: answer generation failed",
                "error": repr(exc),
                "latency_seconds": pruning.latency_seconds,
                "query_input_tokens": query_input_tokens,
                "query_output_tokens": 0,
                "query_total_tokens": query_input_tokens,
                "provider_usage": {},
                "evidence_ids": formatted["included_turn_ids"],
                "top_k": result.get("top_k"),
                "message_range": result.get("message_range"),
                "retrieval_method": result.get("retrieval_method"),
                "temporal_boost": result.get("temporal_boost"),
                "typed_sidecar": result.get("typed_sidecar", False),
                "typed_fact_count": len(result.get("typed_facts", [])),
                "evidence_truncated": formatted["truncated"],
                "provence_pruning": args.provence_pruning,
                "provence_model": args.provence_model if args.provence_pruning else None,
                "provence_threshold": args.provence_threshold if args.provence_pruning else None,
                "provence_input_tokens": pruning.input_tokens,
                "provence_output_tokens": pruning.output_tokens,
                "provence_source_turn_count": pruning.source_turn_count,
                "provence_kept_turn_count": pruning.kept_turn_count,
                "provence_compression_rate": pruning.compression_rate,
                "provence_latency_seconds": pruning.latency_seconds,
            }
            return {"question_id": sample.question_id, "hypothesis": log["hypothesis"]}, log
        log = {
            "question_id": sample.question_id,
            "model": answer.model,
            "hypothesis": answer.hypothesis,
            "latency_seconds": pruning.latency_seconds + answer.latency_seconds,
            "query_input_tokens": answer.prompt_tokens,
            "query_output_tokens": answer.completion_tokens,
            "query_total_tokens": answer.total_tokens,
            "provider_usage": answer.provider_usage,
            "evidence_ids": formatted["included_turn_ids"],
            "top_k": result.get("top_k"),
            "message_range": result.get("message_range"),
            "retrieval_method": result.get("retrieval_method"),
            "temporal_boost": result.get("temporal_boost"),
            "typed_sidecar": result.get("typed_sidecar", False),
            "typed_fact_count": len(result.get("typed_facts", [])),
            "evidence_truncated": formatted["truncated"],
            "provence_pruning": args.provence_pruning,
            "provence_model": args.provence_model if args.provence_pruning else None,
            "provence_threshold": args.provence_threshold if args.provence_pruning else None,
            "provence_input_tokens": pruning.input_tokens,
            "provence_output_tokens": pruning.output_tokens,
            "provence_source_turn_count": pruning.source_turn_count,
            "provence_kept_turn_count": pruning.kept_turn_count,
            "provence_compression_rate": pruning.compression_rate,
            "provence_latency_seconds": pruning.latency_seconds,
        }
        return {"question_id": sample.question_id, "hypothesis": answer.hypothesis}, log

    pending_items = [item for item in qa_items if item[3].question_id not in existing]
    predictions = [{"question_id": qid, "hypothesis": str(log.get("hypothesis", ""))} for qid, log in existing.items()]
    logs = list(existing.values())
    for prediction, log in _iter_parallel(pending_items, answer_one, workers):
        predictions.append(prediction)
        logs.append(log)
        _append_jsonl_record(pred_path, prediction)
        _append_jsonl_record(log_path, log)
    if existing and not pending_items:
        write_predictions(pred_path, predictions)
        write_jsonl(log_path, logs)
    return logs


def _make_provence_pruners(args: argparse.Namespace, count: int) -> list[ProvenceEvidencePruner]:
    pruners = [
        make_provence_pruner(args.provence_model, args.provence_threshold, args.provence_batch_size)
        for _ in range(count)
    ]
    for pruner in pruners:
        pruner.model
    return pruners


def _formatted_evidence_for_answer(
    result: dict[str, Any],
    sample: Any,
    max_evidence_tokens: int | None,
    pruner: ProvenceEvidencePruner | None = None,
) -> dict[str, Any]:
    pruning = ProvencePruningResult([], 0, 0, 0, 0, 0.0, 0.0)
    if result.get("deduped_evidence"):
        evidence_turns = result["deduped_evidence"]
        if pruner is not None:
            pruning = pruner.prune_turns(sample.question, evidence_turns)
            evidence_turns = pruning.turns
        formatted = format_evidence_for_answerer(evidence_turns, sample.question_date, max_evidence_tokens)
        return {
            "text": prepend_typed_fact_pack(formatted.text, result.get("typed_facts", [])),
            "included_turn_ids": formatted.included_turn_ids,
            "truncated": formatted.truncated,
            "pruning": pruning,
        }
    return {
        "text": result.get("typed_augmented_evidence") or result["formatted_evidence"],
        "included_turn_ids": result.get("formatted_evidence_turn_ids", []),
        "truncated": result.get("evidence_truncated", False),
        "pruning": pruning,
    }


def run_judge_stage(args: argparse.Namespace, run_dir: Path, qa_items: list[tuple[int, int, dict[str, Any], Any]]) -> list[dict[str, Any]]:
    predictions = {row["question_id"]: row.get("hypothesis", "") for row in read_jsonl(run_dir / "predictions.jsonl")}
    judge = make_judge(args.judge_model, args.judge_base_url)
    skip_existing = args.skip_existing or args.resume
    existing = _existing_by_id(run_dir / "judge_logs.jsonl") if skip_existing else {}
    log_path = run_dir / "judge_logs.jsonl"
    if not skip_existing:
        log_path.unlink(missing_ok=True)

    def judge_one(item: tuple[int, int, dict[str, Any], Any]) -> dict[str, Any]:
        _conv_idx, _qa_idx, _qa, sample = item
        if sample.question_id in existing:
            return existing[sample.question_id]
        if predictions.get(sample.question_id, "").startswith("ERROR:"):
            return {
                "question_id": sample.question_id,
                "question_type": sample.question_type,
                "category": int(_qa.get("category", 0)),
                "model": judge.model_name,
                "label": "WRONG",
                "score": 0.0,
                "reasoning": "Answer generation failed.",
                "latency_seconds": 0.0,
                "judge_input_tokens": 0,
                "judge_output_tokens": 0,
                "judge_total_tokens": 0,
                "provider_usage": {},
                "raw_response": "",
                "error": "answer_generation_failed",
            }
        try:
            result = judge.judge_locomo(sample.question, sample.answer, predictions.get(sample.question_id, ""))
        except Exception as exc:
            return {
                "question_id": sample.question_id,
                "question_type": sample.question_type,
                "category": int(_qa.get("category", 0)),
                "model": judge.model_name,
                "label": "WRONG",
                "score": 0.0,
                "reasoning": "Judge failed.",
                "latency_seconds": 0.0,
                "judge_input_tokens": 0,
                "judge_output_tokens": 0,
                "judge_total_tokens": 0,
                "provider_usage": {},
                "raw_response": "",
                "error": repr(exc),
            }
        return {
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "category": int(_qa.get("category", 0)),
            "model": result.model,
            "label": result.label,
            "score": result.score,
            "reasoning": result.reasoning,
            "latency_seconds": result.latency_seconds,
            "judge_input_tokens": result.prompt_tokens,
            "judge_output_tokens": result.completion_tokens,
            "judge_total_tokens": result.total_tokens,
            "provider_usage": result.provider_usage,
            "raw_response": result.raw_response,
        }

    pending_items = [item for item in qa_items if item[3].question_id not in existing]
    logs = list(existing.values())
    workers = args.judge_parallelism if args.judge_parallelism is not None else args.parallelism
    for log in _iter_parallel(pending_items, judge_one, workers):
        logs.append(log)
        _append_jsonl_record(log_path, log)
    if existing and not pending_items:
        write_jsonl(log_path, logs)
    return logs


def write_metrics(run_dir: Path, categories: list[int]) -> None:
    retrieval_results = read_jsonl(run_dir / "retrieval_results.jsonl")
    answer_logs = read_jsonl(run_dir / "answer_logs.jsonl")
    judge_logs = read_jsonl(run_dir / "judge_logs.jsonl")
    build_stats = read_jsonl(run_dir / "build_stats.jsonl")
    token_stats = json.loads((run_dir / "token_stats.json").read_text(encoding="utf-8")) if (run_dir / "token_stats.json").exists() else new_token_summary()
    judge_by_id = {row["question_id"]: row for row in judge_logs}
    answer_by_id = {row["question_id"]: row for row in answer_logs}
    valid = [row for row in retrieval_results if int(row.get("category", 0)) in LOCOMO_OVERALL_CATEGORIES]
    breakdown_rows = [row for row in retrieval_results if int(row.get("category", 0)) in categories]
    correct = sum(1 for row in valid if judge_by_id.get(row["question_id"], {}).get("label") == "CORRECT")
    by_type: dict[str, Any] = {}
    for category in categories:
        rows = [row for row in breakdown_rows if int(row.get("category", 0)) == category]
        cat_correct = sum(1 for row in rows if judge_by_id.get(row["question_id"], {}).get("label") == "CORRECT")
        by_type[CATEGORY_NAMES.get(category, str(category))] = {
            "total": len(rows),
            "correct": cat_correct,
            "accuracy": cat_correct / len(rows) if rows else 0.0,
            "avg_query_time_seconds": _mean([_query_time(token_stats, row["question_id"]) for row in rows]),
        }
    retrieval_metrics = _retrieval_overall(valid)
    metrics = {
        "benchmark": "locomo",
        "num_conversations": len(build_stats),
        "num_questions": len(valid),
        "correct": correct,
        "llm_judge_accuracy": correct / len(valid) if valid else 0.0,
        "retrieval": retrieval_metrics,
        "build_time_seconds": token_stats.get("time_stats", {}).get("build_seconds", 0.0),
        "avg_build_time_seconds_per_conversation": _mean([float(row.get("build_time_seconds", 0.0)) for row in build_stats]),
        "query_time_seconds": token_stats.get("time_stats", {}).get("query_seconds", 0.0),
        "avg_query_time_seconds_per_question": _mean([_query_time(token_stats, row["question_id"]) for row in valid]),
        "judge_time_seconds_excluded": token_stats.get("time_stats", {}).get("judge_seconds_excluded", 0.0),
        "avg_judge_time_seconds_per_question": _mean([float(judge_by_id.get(row["question_id"], {}).get("latency_seconds", 0.0)) for row in valid]),
        "build_tokens": token_stats.get("build_tokens", {}),
        "query_tokens": token_stats.get("query_tokens", {}),
        "judge_tokens_excluded": token_stats.get("judge_tokens", {}),
        "num_answered": len(answer_by_id),
        "num_judged": len(judge_by_id),
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "metrics_by_type.json", by_type)


def write_topk_summary(run_dir: Path, cutoffs: list[int]) -> None:
    accuracy_by_cutoff = {}
    by_type = {}
    for cutoff in cutoffs:
        label = f"top_{cutoff}"
        cutoff_dir = run_dir / label
        metrics_path = cutoff_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        accuracy = float(metrics.get("llm_judge_accuracy", 0.0))
        accuracy_by_cutoff[label] = {
            "cutoff": cutoff,
            "correct": metrics.get("correct", 0),
            "total": metrics.get("num_questions", 0),
            "accuracy": accuracy,
            "accuracy_percent": accuracy * 100,
            "build_time_seconds": metrics.get("build_time_seconds", 0.0),
            "query_time_seconds": metrics.get("query_time_seconds", 0.0),
            "judge_time_seconds_excluded": metrics.get("judge_time_seconds_excluded", 0.0),
            "metrics_path": f"{label}/metrics.json",
        }
        by_type[label] = json.loads((cutoff_dir / "metrics_by_type.json").read_text(encoding="utf-8")) if (cutoff_dir / "metrics_by_type.json").exists() else {}
    payload = {
        "benchmark": "locomo",
        "top_k_list": cutoffs,
        "accuracy_by_cutoff": accuracy_by_cutoff,
    }
    write_json(run_dir / "topk_metrics.json", payload)
    write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "metrics_by_type.json", by_type)
    _write_topk_markdown(run_dir / "topk_metrics.md", accuracy_by_cutoff)


def _prepare_cutoff_run_dir(parent_dir: Path, cutoff_dir: Path, cutoff: int, max_evidence_tokens: int | None) -> None:
    rows = [_slice_retrieval_result(row, cutoff, max_evidence_tokens) for row in read_jsonl(parent_dir / "retrieval_results.jsonl")]
    write_jsonl(cutoff_dir / "retrieval_results.jsonl", rows)
    source = parent_dir / "build_stats.jsonl"
    if source.exists():
        shutil.copyfile(source, cutoff_dir / "build_stats.jsonl")
    config_path = parent_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["top_k"] = cutoff
        config["parent_run_dir"] = str(parent_dir)
        write_json(cutoff_dir / "config.json", config)


def _slice_retrieval_result(row: dict[str, Any], cutoff: int, max_evidence_tokens: int | None) -> dict[str, Any]:
    sliced = copy.deepcopy(row)
    matched_turns = sliced.get("matched_turns", [])[:cutoff]
    evidence_windows = sliced.get("evidence_windows", [])[:cutoff]
    deduped_evidence = _dedupe_and_sort_windows(evidence_windows)
    formatted = format_evidence_for_answerer(deduped_evidence, sliced.get("question_date", ""), max_evidence_tokens)
    sliced["top_k"] = cutoff
    sliced["matched_turns"] = matched_turns
    sliced["evidence_windows"] = evidence_windows
    sliced["deduped_evidence"] = deduped_evidence
    sliced["formatted_evidence"] = formatted.text
    sliced["formatted_evidence_turn_ids"] = formatted.included_turn_ids
    sliced["evidence_truncated"] = formatted.truncated
    sliced["evidence_truncate_strategy"] = formatted.truncate_strategy
    sliced["metrics"] = _retrieval_metrics_from_row(sliced, matched_turns, deduped_evidence, formatted.token_count)
    return sliced


def _retrieval_metrics_from_row(row: dict[str, Any], matched_turns: list[dict[str, Any]], deduped_evidence: list[dict[str, Any]], evidence_token_count: int) -> dict[str, Any]:
    skipped = not row.get("answer_session_ids")
    answer_session_ids = {str(session_id) for session_id in row.get("answer_session_ids", [])}
    matched_session_ids = {str(turn.get("session_id", "")) for turn in matched_turns}
    deduped_session_ids = {str(turn.get("session_id", "")) for turn in deduped_evidence}
    turn_hit = any(turn.get("has_answer") for turn in matched_turns) if not skipped else None
    expanded_turn_hit = any(turn.get("has_answer") for turn in deduped_evidence) if not skipped else None
    return {
        "skipped": skipped,
        "session_recall_at_k": bool(answer_session_ids & matched_session_ids) if not skipped else None,
        "turn_recall_at_k": turn_hit,
        "expanded_turn_recall_at_k": expanded_turn_hit,
        "turn_or_expanded_recall_at_k": bool(turn_hit or expanded_turn_hit) if not skipped else None,
        "evidence_token_count": evidence_token_count,
        "num_matched_turns": len(matched_turns),
        "num_deduped_turns": len(deduped_evidence),
        "num_sessions_recalled": len(answer_session_ids & matched_session_ids) if not skipped else 0,
        "num_deduped_sessions": len(deduped_session_ids),
    }


def _load_cutoff_token_stats(parent_dir: Path, cutoff_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    cutoff_path = cutoff_dir / "token_stats.json"
    if (args.resume or args.skip_existing) and cutoff_path.exists():
        return json.loads(cutoff_path.read_text(encoding="utf-8"))
    parent_path = parent_dir / "token_stats.json"
    return json.loads(parent_path.read_text(encoding="utf-8")) if parent_path.exists() else new_token_summary()


def _write_topk_markdown(path: Path, accuracy_by_cutoff: dict[str, Any]) -> None:
    lines = [
        "# LoCoMo Top-k Metrics",
        "",
        "| Cutoff | Correct / Total | Accuracy | Query Time (s) | Judge Time Excluded (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, data in accuracy_by_cutoff.items():
        lines.append(
            f"| {label} | {data['correct']} / {data['total']} | {data['accuracy_percent']:.2f}% | "
            f"{float(data['query_time_seconds']):.2f} | {float(data['judge_time_seconds_excluded']):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _with_dynamic_labels(turn: dict[str, Any], evidence_ids: set[str], answer_sessions: set[str]) -> dict[str, Any]:
    turn["has_answer"] = str(turn.get("locomo_dia_id", "")) in evidence_ids
    turn["answer_session_label"] = str(turn.get("session_id", "")) in answer_sessions
    return turn


def _store_dir(run_dir: Path, conv_idx: int) -> Path:
    return run_dir / "stores" / safe_filename(f"conv{conv_idx}")


def _check_store(store_dir: Path, model_name: str, chunk_mode: str) -> None:
    stats = json.loads((store_dir / "store_stats.json").read_text(encoding="utf-8"))
    if stats.get("embedder_model") != model_name:
        raise ValueError(f"{store_dir} was built with {stats.get('embedder_model')!r}, but retrieval is using {model_name!r}.")
    _check_chunk_mode(stats, chunk_mode, store_dir)


def _check_chunk_mode(stats: dict[str, Any], chunk_mode: str, store_dir: Path) -> None:
    built_with = stats.get("chunk_mode", "turn")
    if built_with != chunk_mode:
        raise ValueError(f"{store_dir} was built with chunk_mode {built_with!r}; re-run build with --force-rebuild.")


def _existing_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row["question_id"]: row for row in read_jsonl(path) if "question_id" in row}


def _parallel_map(items: list[Any], fn: Any, workers: int) -> tuple[list[Any], list[Any]]:
    if workers <= 1:
        outputs = [fn(item) for item in items]
    else:
        ordered = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fn, item): idx for idx, item in enumerate(items)}
            for future in as_completed(futures):
                ordered.append((futures[future], future.result()))
        outputs = [value for _, value in sorted(ordered, key=lambda item: item[0])]
    if outputs and isinstance(outputs[0], tuple):
        return [output[0] for output in outputs], [output[1] for output in outputs]
    return [], outputs


def _iter_parallel(items: list[Any], fn: Any, workers: int) -> Any:
    if workers <= 1:
        for item in items:
            yield fn(item)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fn, item) for item in items]
        for future in as_completed(futures):
            yield future.result()


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _query_time(token_stats: dict[str, Any], question_id: str) -> float:
    return float(token_stats.get("per_question", {}).get(question_id, {}).get("query_time_seconds", 0.0))


def _retrieval_overall(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row.get("metrics", {}) for row in results if not row.get("metrics", {}).get("skipped")]
    return {
        "session_recall_at_k": _mean([1.0 if m.get("session_recall_at_k") else 0.0 for m in metrics]),
        "turn_recall_at_k": _mean([1.0 if m.get("turn_recall_at_k") else 0.0 for m in metrics]),
        "expanded_turn_recall_at_k": _mean([1.0 if m.get("expanded_turn_recall_at_k") else 0.0 for m in metrics]),
        "avg_evidence_tokens": _mean([float(m.get("evidence_token_count", 0.0)) for m in metrics]),
        "avg_num_turns": _mean([float(m.get("num_deduped_turns", 0.0)) for m in metrics]),
        "avg_num_sessions": _mean([float(m.get("num_deduped_sessions", 0.0)) for m in metrics]),
    }


def _load_or_new_token_stats(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "token_stats.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else new_token_summary()


def _reset_build(token_stats: dict[str, Any]) -> None:
    token_stats["build_tokens"] = {"embedding_input_tokens": 0, "embedding_provider_tokens": 0, "llm_input_tokens": 0, "llm_output_tokens": 0, "llm_total_tokens": 0}
    token_stats.setdefault("time_stats", {})["build_seconds"] = 0.0
    token_stats["method_cost_tokens"]["build_embedding_input_tokens"] = 0


def _reset_retrieval(token_stats: dict[str, Any]) -> None:
    token_stats["retrieval_embedding_tokens"] = {"input_tokens": 0}


def _reset_query(token_stats: dict[str, Any]) -> None:
    token_stats["query_tokens"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    token_stats.setdefault("time_stats", {})["query_seconds"] = 0.0
    token_stats["method_cost_tokens"]["query_total_tokens"] = 0


def _reset_judge(token_stats: dict[str, Any]) -> None:
    token_stats["judge_tokens"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    token_stats.setdefault("time_stats", {})["judge_seconds_excluded"] = 0.0
    token_stats["method_cost_tokens"]["judge_total_tokens_excluded"] = 0


def _config_dict(args: argparse.Namespace, conversations: list[int], categories: list[int], qa_items: list[Any]) -> dict[str, Any]:
    return {
        "data": args.data,
        "run_id": args.run_id,
        "conversations": conversations,
        "categories": categories,
        "top_k": args.top_k,
        "message_range": args.message_range,
        "chunk_mode": args.chunk_mode,
        "retrieval_method": args.retrieval_method,
        "temporal_boost": args.temporal_boost,
        "embedding_model": args.embedding_model,
        "embedding_backend": args.embedding_backend,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "question_limit": args.question_limit,
        "parallelism": args.parallelism,
        "answer_parallelism": args.answer_parallelism,
        "judge_parallelism": args.judge_parallelism,
        "cutoff_parallelism": args.cutoff_parallelism,
        "mode": args.mode,
        "max_evidence_tokens": args.max_evidence_tokens,
        "provence_pruning": args.provence_pruning,
        "provence_model": args.provence_model,
        "provence_threshold": args.provence_threshold,
        "provence_batch_size": args.provence_batch_size,
        "typed_sidecar": args.typed_sidecar,
        "output_dir": args.output_dir,
        "num_selected_questions": len(qa_items),
    }


def _top_k_cutoffs(args: argparse.Namespace) -> list[int]:
    values = parse_int_list(args.top_k_list, [args.top_k]) if args.top_k_list else [args.top_k]
    return sorted(dict.fromkeys(values))


def _run_stage_subprocess(args: argparse.Namespace, mode: str, keep_cuda: bool) -> None:
    env = dict(os.environ)
    env["MEMORY_BASELINE_LOCOMO_SPLIT_CHILD"] = "1"
    if not keep_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([sys.executable, "-m", "memory_baseline.run_locomo", *_subprocess_args(args, mode)], check=True, env=env)


def _subprocess_args(args: argparse.Namespace, mode: str) -> list[str]:
    values: list[str] = [
        "--data",
        args.data,
        "--run-id",
        args.run_id,
        "--conversations",
        args.conversations,
        "--categories",
        args.categories,
        "--top-k",
        str(args.top_k),
        "--message-range",
        str(args.message_range),
        "--chunk-mode",
        args.chunk_mode,
        "--retrieval-method",
        args.retrieval_method,
        "--temporal-boost",
        str(args.temporal_boost),
        "--embedding-model",
        args.embedding_model,
        "--embedding-backend",
        args.embedding_backend,
        "--parallelism",
        str(args.parallelism),
        "--mode",
        mode,
        "--output-dir",
        args.output_dir,
    ]
    for option, value in [
        ("--top-k-list", args.top_k_list),
        ("--embedding-base-url", args.embedding_base_url),
        ("--answer-model", args.answer_model),
        ("--answer-base-url", args.answer_base_url),
        ("--judge-model", args.judge_model),
        ("--judge-base-url", args.judge_base_url),
        ("--question-limit", args.question_limit),
        ("--answer-parallelism", args.answer_parallelism),
        ("--judge-parallelism", args.judge_parallelism),
        ("--cutoff-parallelism", args.cutoff_parallelism),
        ("--max-evidence-tokens", args.max_evidence_tokens),
        ("--provence-model", args.provence_model),
        ("--provence-threshold", args.provence_threshold),
        ("--provence-batch-size", args.provence_batch_size),
    ]:
        if value is not None:
            values.extend([option, str(value)])
    for flag in ["resume", "force_rebuild", "skip_existing", "provence_pruning", "typed_sidecar"]:
        if getattr(args, flag):
            values.append("--" + flag.replace("_", "-"))
    return values


def _backend_arg(value: str | None) -> str | None:
    return None if value in {None, "auto"} else value


def main(argv: list[str] | None = None) -> None:
    result = run_pipeline(parse_args(argv))
    print(f"run_dir={result['run_dir']} num_conversations={result['num_conversations']} num_questions={result['num_questions']}")


if __name__ == "__main__":
    main()
