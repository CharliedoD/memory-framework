from __future__ import annotations

import argparse
import copy
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any

from memory_baseline.cli.judge_longmemeval import _eval_row, _metrics
from memory_baseline.core.env import load_project_env
from memory_baseline.generation.answerer import answer_focus_for_question_type, make_answerer
from memory_baseline.generation.evidence_compiler import make_evidence_compiler
from memory_baseline.generation.judge import make_judge
from memory_baseline.generation.provence_pruner import (
    DEFAULT_PROVENCE_MODEL,
    ProvenceEvidencePruner,
    ProvencePruningResult,
    make_provence_pruner,
)
from memory_baseline.indexing.embedder import embed_texts_cached, make_embedder
from memory_baseline.evaluation.error_analysis import write_error_analysis
from memory_baseline.evaluation.retrieval import aggregate_retrieval_results
from memory_baseline.core.io import read_predictions, write_predictions
from memory_baseline.data.longmemeval import load_question_ids, load_samples, parse_question_types
from memory_baseline.retrieval.dense import DenseRetriever, _dedupe_and_sort_windows
from memory_baseline.retrieval.formatter import format_evidence_for_answerer
from memory_baseline.retrieval.typed_facts import prepend_typed_fact_pack
from memory_baseline.core.schemas import LongMemEvalSample
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
from memory_baseline.core.utils import ensure_dir, estimate_tokens, id_key, read_jsonl, write_json, write_jsonl
from memory_baseline.indexing.vector_store import QuestionStore, build_question_store, question_store_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/longmemeval_s_cleaned.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-k-list")
    parser.add_argument("--message-range", type=int, default=2)
    parser.add_argument("--chunk-mode", choices=["turn"], default="turn")
    parser.add_argument("--embedding-text-mode", choices=["metadata_content", "content"], default="content")
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
    parser.add_argument("--question-ids")
    parser.add_argument("--question-types")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--answer-parallelism", type=int)
    parser.add_argument("--judge-parallelism", type=int)
    parser.add_argument("--cutoff-parallelism", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=["build", "retrieve", "query", "answer", "judge", "eval-retrieval", "eval", "index", "api", "full"],
        default="full",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-evidence-tokens", type=int)
    parser.add_argument("--disable-evidence-compiler", action="store_true")
    parser.add_argument("--provence-pruning", action="store_true")
    parser.add_argument("--provence-model", default=DEFAULT_PROVENCE_MODEL)
    parser.add_argument("--provence-threshold", type=float, default=0.1)
    parser.add_argument("--provence-batch-size", type=int, default=32)
    parser.add_argument("--typed-sidecar", action="store_true")
    parser.add_argument("--output-dir", default="runs")
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    cutoffs = _top_k_cutoffs(args)
    if len(cutoffs) > 1:
        return run_topk_pipeline(args, cutoffs)

    run_dir = ensure_dir(Path(args.output_dir) / args.run_id)
    samples = _load_samples_for_run(args)
    write_json(run_dir / "config.json", _config_dict(args, samples))
    token_stats = _load_or_new_token_stats(run_dir)

    if args.mode in {"build", "index", "full"}:
        build_stats = run_build_stage(args, run_dir, samples)
        _reset_build_tokens(token_stats)
        for stat in build_stats:
            add_build_tokens(
                token_stats,
                stat["question_id"],
                stat.get("build_embedding_input_tokens", 0),
                stat.get("build_embedding_provider_tokens", 0),
            )
            add_build_time(token_stats, stat["question_id"], stat.get("build_time_seconds", 0.0))
        write_json(run_dir / "token_stats.json", token_stats)

    if args.mode in {"retrieve", "query", "index", "full"}:
        retrieval_results = run_retrieve_stage(args, run_dir, samples)
        _reset_retrieval_embedding_tokens(token_stats)
        for result in retrieval_results:
            add_retrieval_embedding_tokens(token_stats, result["question_id"], result.get("query_embedding_tokens", 0))
        write_json(run_dir / "token_stats.json", token_stats)

    if args.mode in {"eval-retrieval", "eval", "index", "full"}:
        retrieval_results = read_jsonl(run_dir / "retrieval_results.jsonl")
        overall, by_type = aggregate_retrieval_results(retrieval_results)
        write_json(run_dir / "metrics.json", overall)
        write_json(run_dir / "metrics_by_type.json", by_type)

    if args.mode in {"query", "answer", "api", "full"}:
        answer_logs = run_answer_stage(args, run_dir, samples)
        _reset_query_tokens(token_stats)
        for log in answer_logs:
            query_time = float(log.get("retrieval_latency_seconds", 0.0)) + float(log.get("latency_seconds", 0.0))
            add_query_tokens(
                token_stats,
                log["question_id"],
                log.get("query_input_tokens", 0),
                log.get("query_output_tokens", 0),
            )
            add_query_time(token_stats, log["question_id"], query_time)
        write_json(run_dir / "token_stats.json", token_stats)
        write_error_analysis(
            samples,
            read_jsonl(run_dir / "retrieval_results.jsonl"),
            run_dir / "predictions.jsonl",
            token_stats,
            run_dir / "error_analysis.jsonl",
            autoeval_log=Path(str(run_dir / "predictions.jsonl") + ".log"),
        )

    if args.mode in {"judge", "api", "full"}:
        judge_logs = run_judge_stage(args, run_dir, samples)
        _reset_judge_tokens(token_stats)
        for log in judge_logs:
            add_judge_tokens(token_stats, log["question_id"], log.get("judge_input_tokens", 0), log.get("judge_output_tokens", 0))
            add_judge_time(token_stats, log["question_id"], log.get("latency_seconds", 0.0))
        write_json(run_dir / "token_stats.json", token_stats)
        write_final_metrics(run_dir, samples)

    if args.mode == "eval":
        write_final_metrics(run_dir, samples)

    return {"run_dir": str(run_dir), "num_samples": len(samples)}


def run_topk_pipeline(args: argparse.Namespace, cutoffs: list[int]) -> dict[str, Any]:
    run_dir = ensure_dir(Path(args.output_dir) / args.run_id)
    samples = _load_samples_for_run(args)
    max_args = argparse.Namespace(**vars(args))
    max_args.top_k = max(cutoffs)
    config = _config_dict(max_args, samples)
    config["top_k_list"] = cutoffs
    write_json(run_dir / "config.json", config)
    token_stats = _load_or_new_token_stats(run_dir)

    if args.mode in {"build", "index", "full"}:
        build_stats = run_build_stage(max_args, run_dir, samples)
        _reset_build_tokens(token_stats)
        for stat in build_stats:
            add_build_tokens(
                token_stats,
                stat["question_id"],
                stat.get("build_embedding_input_tokens", 0),
                stat.get("build_embedding_provider_tokens", 0),
            )
            add_build_time(token_stats, stat["question_id"], stat.get("build_time_seconds", 0.0))
        write_json(run_dir / "token_stats.json", token_stats)

    if args.mode in {"retrieve", "query", "index", "full"}:
        retrieval_results = run_retrieve_stage(max_args, run_dir, samples)
        _reset_retrieval_embedding_tokens(token_stats)
        for result in retrieval_results:
            add_retrieval_embedding_tokens(token_stats, result["question_id"], result.get("query_embedding_tokens", 0))
        write_json(run_dir / "token_stats.json", token_stats)

    if args.mode in {"retrieve", "query", "eval-retrieval", "eval", "index", "full", "answer", "judge", "api"}:
        _prepare_cutoff_run_dirs(run_dir, cutoffs, args.max_evidence_tokens)

    if args.mode in {"query", "answer", "judge", "eval", "api", "full"}:
        def run_cutoff(cutoff: int) -> int:
            cutoff_dir = ensure_dir(run_dir / f"top_{cutoff}")
            cutoff_args = argparse.Namespace(**vars(args))
            cutoff_args.top_k = cutoff
            cutoff_token_stats = _load_cutoff_token_stats(run_dir, cutoff_dir, args)
            if args.mode in {"query", "answer", "api", "full"}:
                answer_logs = run_answer_stage(cutoff_args, cutoff_dir, samples)
                _reset_query_tokens(cutoff_token_stats)
                retrieval_by_id = {row["question_id"]: row for row in read_jsonl(cutoff_dir / "retrieval_results.jsonl")}
                for log in answer_logs:
                    query_time = float(retrieval_by_id.get(log["question_id"], {}).get("retrieval_latency_seconds", 0.0)) + float(log.get("latency_seconds", 0.0))
                    add_query_tokens(cutoff_token_stats, log["question_id"], log.get("query_input_tokens", 0), log.get("query_output_tokens", 0))
                    add_query_time(cutoff_token_stats, log["question_id"], query_time)
                write_json(cutoff_dir / "token_stats.json", cutoff_token_stats)
            if args.mode in {"judge", "api", "full"}:
                judge_logs = run_judge_stage(cutoff_args, cutoff_dir, samples)
                _reset_judge_tokens(cutoff_token_stats)
                for log in judge_logs:
                    add_judge_tokens(cutoff_token_stats, log["question_id"], log.get("judge_input_tokens", 0), log.get("judge_output_tokens", 0))
                    add_judge_time(cutoff_token_stats, log["question_id"], log.get("latency_seconds", 0.0))
                write_json(cutoff_dir / "token_stats.json", cutoff_token_stats)
            if args.mode in {"judge", "eval", "api", "full"}:
                write_final_metrics(cutoff_dir, samples)
            return cutoff

        for _cutoff in _iter_parallel(cutoffs, run_cutoff, args.cutoff_parallelism):
            pass

    if args.mode in {"retrieve", "eval-retrieval", "index"}:
        for cutoff in cutoffs:
            cutoff_dir = run_dir / f"top_{cutoff}"
            retrieval_metrics, retrieval_by_type = aggregate_retrieval_results(read_jsonl(cutoff_dir / "retrieval_results.jsonl"))
            write_json(cutoff_dir / "metrics.json", retrieval_metrics)
            write_json(cutoff_dir / "metrics_by_type.json", retrieval_by_type)

    return {"run_dir": str(run_dir), "num_samples": len(samples), "top_k_list": cutoffs}


def sample_memory_id(sample: LongMemEvalSample) -> str:
    return sample.question_id


def group_by_memory(
    indexed_samples: list[tuple[int, LongMemEvalSample]],
) -> list[tuple[str, list[tuple[int, LongMemEvalSample]]]]:
    groups: OrderedDict[str, list[tuple[int, LongMemEvalSample]]] = OrderedDict()
    for index, sample in indexed_samples:
        groups.setdefault(sample_memory_id(sample), []).append((index, sample))
    return list(groups.items())


def run_build_stage(args: argparse.Namespace, run_dir: Path, samples: list[LongMemEvalSample]) -> list[dict[str, Any]]:
    embedder = make_embedder(args.embedding_model, args.embedding_base_url, backend=_backend_arg(args.embedding_backend))
    skip_existing = args.skip_existing or args.resume
    workers = max(1, args.parallelism)
    indexed_samples = list(enumerate(samples))
    groups = group_by_memory(indexed_samples)

    def build_group(memory_id: str, group: list[tuple[int, LongMemEvalSample]]) -> tuple[str, dict[str, Any]]:
        sample = group[0][1]
        stat = build_question_store(
            sample,
            run_dir,
            embedder,
            cache_root=Path(".cache") / "embeddings",
            force_rebuild=args.force_rebuild,
            skip_existing=skip_existing,
            chunk_mode=args.chunk_mode,
            embedding_text_mode=args.embedding_text_mode,
            typed_sidecar=args.typed_sidecar,
        )
        return memory_id, stat

    stats_by_id: dict[str, dict[str, Any]] = {}
    if workers <= 1 or len(groups) <= 1:
        for done, (memory_id, group) in enumerate(groups, start=1):
            _memory_id, stat = build_group(memory_id, group)
            for _row, sample in group:
                stats_by_id[sample.question_id] = stat
            first_index = group[0][0] + 1
            print(
                f"[build {done}/{len(groups)}] sample={first_index}/{len(samples)} "
                f"memory {memory_id} turns={stat.get('num_turns')}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(build_group, memory_id, group): (memory_id, group) for memory_id, group in groups}
            for done, future in enumerate(as_completed(futures), start=1):
                memory_id, group = futures[future]
                _memory_id, stat = future.result()
                for _row, sample in group:
                    stats_by_id[sample.question_id] = stat
                first_index = group[0][0] + 1
                print(
                    f"[build {done}/{len(groups)}] sample={first_index}/{len(samples)} "
                    f"memory {memory_id} turns={stat.get('num_turns')}",
                    flush=True,
                )

    stats = [stats_by_id[sample.question_id] for sample in samples]
    write_jsonl(run_dir / "build_stats.jsonl", stats)
    return stats


def run_retrieve_stage(args: argparse.Namespace, run_dir: Path, samples: list[LongMemEvalSample]) -> list[dict[str, Any]]:
    embedder = make_embedder(args.embedding_model, args.embedding_base_url, backend=_backend_arg(args.embedding_backend))
    query_batch = embed_texts_cached(embedder, [sample.question for sample in samples], cache_root=Path(".cache") / "embeddings")
    query_tokens = [estimate_tokens(sample.question) for sample in samples]
    workers = max(1, args.parallelism)
    indexed_samples = list(enumerate(samples))
    groups = group_by_memory(indexed_samples)

    def retrieve_group(memory_id: str, group: list[tuple[int, LongMemEvalSample]]) -> tuple[str, list[dict[str, Any]]]:
        store_dir = question_store_dir(run_dir, memory_id)
        if not (store_dir / "embeddings.npy").exists():
            raise FileNotFoundError(f"Missing store for {memory_id}; run --mode build first.")
        _check_store(store_dir, embedder.model_name, args.chunk_mode, args.embedding_text_mode)
        retriever = DenseRetriever(QuestionStore(store_dir), embedder, cache_root=Path(".cache") / "embeddings")
        results = []
        for row, sample in group:
            results.append(
                retriever.retrieve(
                    sample,
                    top_k=args.top_k,
                    message_range=args.message_range,
                    max_evidence_tokens=args.max_evidence_tokens,
                    retrieval_method=args.retrieval_method,
                    temporal_boost=args.temporal_boost,
                    query_vector=query_batch.vectors[row],
                    query_embedding_tokens=query_tokens[row],
                    typed_sidecar=args.typed_sidecar,
                )
            )
        return memory_id, results

    results_by_id: dict[str, dict[str, Any]] = {}
    if workers <= 1 or len(groups) <= 1:
        for done, (memory_id, group) in enumerate(groups, start=1):
            _memory_id, group_results = retrieve_group(memory_id, group)
            for result in group_results:
                results_by_id[result["question_id"]] = result
            first_index = group[0][0] + 1
            print(f"[retrieve {done}/{len(groups)}] sample={first_index}/{len(samples)} memory {memory_id}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(retrieve_group, memory_id, group): (memory_id, group) for memory_id, group in groups}
            for done, future in enumerate(as_completed(futures), start=1):
                memory_id, group = futures[future]
                _memory_id, group_results = future.result()
                for result in group_results:
                    results_by_id[result["question_id"]] = result
                first_index = group[0][0] + 1
                print(f"[retrieve {done}/{len(groups)}] sample={first_index}/{len(samples)} memory {memory_id}", flush=True)

    results = [results_by_id[sample.question_id] for sample in samples]
    write_jsonl(run_dir / "retrieval_results.jsonl", results)
    return results


def run_answer_stage(args: argparse.Namespace, run_dir: Path, samples: list[LongMemEvalSample]) -> list[dict[str, Any]]:
    retrieval_results = read_jsonl(run_dir / "retrieval_results.jsonl")
    retrieval_by_id = {result["question_id"]: result for result in retrieval_results}
    answerer = make_answerer(args.answer_model, args.answer_base_url)
    compiler = None if args.disable_evidence_compiler else make_evidence_compiler(args.answer_model, args.answer_base_url)
    workers = args.answer_parallelism if args.answer_parallelism is not None else args.parallelism
    pruners = _make_provence_pruners(args, max(1, workers)) if args.provence_pruning else None
    pruner_queue = Queue()
    for pruner in pruners or []:
        pruner_queue.put(pruner)
    skip_existing = args.skip_existing or args.resume
    existing_predictions = read_predictions(run_dir / "predictions.jsonl") if skip_existing else {}
    existing_logs = {
        record["question_id"]: record
        for record in (read_jsonl(run_dir / "answer_logs.jsonl") if skip_existing else [])
        if "question_id" in record
    }
    pred_path = run_dir / "predictions.jsonl"
    log_path = run_dir / "answer_logs.jsonl"
    if not skip_existing:
        pred_path.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)

    def answer_one(sample: LongMemEvalSample) -> tuple[dict[str, str], dict[str, Any]]:
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
        compiled = None
        evidence_text = formatted["text"]
        if compiler is not None and sample.question_type in {"knowledge-update", "temporal-reasoning"}:
            compiled = compiler.compile(sample.question_date, formatted["text"], sample.question, sample.question_type)
            evidence_text = compiled.text
        answer = answerer.answer(sample.question_date, evidence_text, sample.question, sample.question_type)
        evidence_ids = formatted["included_turn_ids"]
        pruning = formatted["pruning"]
        compiler_input_tokens = compiled.prompt_tokens if compiled is not None else 0
        compiler_output_tokens = compiled.completion_tokens if compiled is not None else 0
        compiler_total_tokens = compiled.total_tokens if compiled is not None else 0
        query_input_tokens = compiler_input_tokens + answer.prompt_tokens
        query_output_tokens = compiler_output_tokens + answer.completion_tokens
        log = {
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "model": answer.model,
            "latency_seconds": pruning.latency_seconds + (compiled.latency_seconds if compiled is not None else 0.0) + answer.latency_seconds,
            "retrieval_latency_seconds": result.get("retrieval_latency_seconds", 0.0),
            "query_input_tokens": query_input_tokens,
            "query_output_tokens": query_output_tokens,
            "query_total_tokens": query_input_tokens + query_output_tokens,
            "evidence_compiler_disabled": args.disable_evidence_compiler,
            "compiler_model": compiled.model if compiled is not None else None,
            "compiler_input_tokens": compiler_input_tokens,
            "compiler_output_tokens": compiler_output_tokens,
            "compiler_total_tokens": compiler_total_tokens,
            "compiler_provider_usage": compiled.provider_usage if compiled is not None else {},
            "compiled_evidence": compiled.text if compiled is not None else None,
            "provence_pruning": args.provence_pruning,
            "provence_model": args.provence_model if args.provence_pruning else None,
            "provence_threshold": args.provence_threshold if args.provence_pruning else None,
            "provence_input_tokens": pruning.input_tokens,
            "provence_output_tokens": pruning.output_tokens,
            "provence_source_turn_count": pruning.source_turn_count,
            "provence_kept_turn_count": pruning.kept_turn_count,
            "provence_compression_rate": pruning.compression_rate,
            "provence_latency_seconds": pruning.latency_seconds,
            "answer_input_tokens": answer.prompt_tokens,
            "answer_output_tokens": answer.completion_tokens,
            "answer_total_tokens": answer.total_tokens,
            "provider_usage": answer.provider_usage,
            "evidence_ids": evidence_ids,
            "top_k": result.get("top_k"),
            "message_range": result.get("message_range"),
            "retrieval_method": result.get("retrieval_method"),
            "temporal_boost": result.get("temporal_boost"),
            "typed_sidecar": result.get("typed_sidecar", False),
            "typed_fact_count": len(result.get("typed_facts", [])),
            "evidence_truncated": formatted["truncated"],
            "answer_focus": answer_focus_for_question_type(sample.question_type),
        }
        return {"question_id": sample.question_id, "hypothesis": answer.hypothesis}, log

    def answer_group(
        memory_id: str,
        group: list[tuple[int, LongMemEvalSample]],
    ) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
        group_predictions = []
        group_logs = []
        for _row, sample in group:
            prediction, log = answer_one(sample)
            group_predictions.append(prediction)
            group_logs.append(log)
        return memory_id, group_predictions, group_logs

    indexed_samples = list(enumerate(samples))
    groups = group_by_memory(indexed_samples)
    pending_groups = [
        (memory_id, [(row, sample) for row, sample in group if sample.question_id not in existing_predictions])
        for memory_id, group in groups
    ]
    pending_groups = [(memory_id, group) for memory_id, group in pending_groups if group]
    predictions: list[dict[str, str]] = [
        {"question_id": question_id, "hypothesis": hypothesis}
        for question_id, hypothesis in existing_predictions.items()
    ]
    logs: list[dict[str, Any]] = list(existing_logs.values())
    if workers > 1 and len(pending_groups) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(answer_group, memory_id, group): (memory_id, group) for memory_id, group in pending_groups}
            for done, future in enumerate(as_completed(futures), start=1):
                memory_id, group = futures[future]
                _memory_id, group_predictions, group_logs = future.result()
                for prediction, log in zip(group_predictions, group_logs):
                    predictions.append(prediction)
                    logs.append(log)
                    _append_jsonl_record(pred_path, prediction)
                    _append_jsonl_record(log_path, log)
                first_index = group[0][0] + 1
                print(f"[answer {done}/{len(pending_groups)}] sample={first_index}/{len(samples)} memory {memory_id}", flush=True)
    else:
        for done, (memory_id, group) in enumerate(pending_groups, start=1):
            _memory_id, group_predictions, group_logs = answer_group(memory_id, group)
            for prediction, log in zip(group_predictions, group_logs):
                predictions.append(prediction)
                logs.append(log)
                _append_jsonl_record(pred_path, prediction)
                _append_jsonl_record(log_path, log)
            first_index = group[0][0] + 1
            print(f"[answer {done}/{len(pending_groups)}] sample={first_index}/{len(samples)} memory {memory_id}", flush=True)
    if existing_predictions and not pending_groups:
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


def run_judge_stage(args: argparse.Namespace, run_dir: Path, samples: list[LongMemEvalSample]) -> list[dict[str, Any]]:
    sample_by_id = {sample.question_id: sample for sample in samples}
    predictions = read_predictions(run_dir / "predictions.jsonl")
    judge = make_judge(args.judge_model, args.judge_base_url)
    skip_existing = args.skip_existing or args.resume
    existing = {
        row["question_id"]: row
        for row in (read_jsonl(run_dir / "judge_logs.jsonl") if skip_existing else [])
        if "question_id" in row
    }
    log_path = run_dir / "judge_logs.jsonl"
    eval_path = run_dir / f"predictions.jsonl.eval-results-{judge.model_name}"
    if not skip_existing:
        log_path.unlink(missing_ok=True)
        eval_path.unlink(missing_ok=True)

    def judge_one(question_id: str) -> dict[str, Any]:
        sample = sample_by_id[question_id]
        result = judge.judge_longmemeval(
            sample.question_type,
            sample.question,
            sample.answer,
            predictions[question_id],
            abstention="_abs" in question_id,
        )
        return {
            "question_id": question_id,
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

    pending = [question_id for question_id in predictions if question_id in sample_by_id and question_id not in existing]
    logs = dict(existing)
    workers = args.judge_parallelism if args.judge_parallelism is not None else args.parallelism
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(judge_one, question_id): question_id for question_id in pending}
            for done, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                logs[row["question_id"]] = row
                _append_jsonl_record(log_path, row)
                _append_jsonl_record(eval_path, _eval_row(sample_by_id[row["question_id"]], predictions[row["question_id"]], row))
                print(f"[judge {done}/{len(pending)}] {row['question_id']} label={row['label']}", flush=True)
    else:
        for done, question_id in enumerate(pending, start=1):
            row = judge_one(question_id)
            logs[row["question_id"]] = row
            _append_jsonl_record(log_path, row)
            _append_jsonl_record(eval_path, _eval_row(sample_by_id[row["question_id"]], predictions[row["question_id"]], row))
            print(f"[judge {done}/{len(pending)}] {row['question_id']} label={row['label']}", flush=True)

    ordered_logs = [logs[question_id] for question_id in predictions if question_id in logs]
    if existing and not pending:
        write_jsonl(log_path, ordered_logs)
        write_jsonl(
            eval_path,
            [_eval_row(sample_by_id[row["question_id"]], predictions[row["question_id"]], row) for row in ordered_logs],
        )
    return ordered_logs


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _load_samples_for_run(args: argparse.Namespace) -> list[LongMemEvalSample]:
    samples = load_samples(
        args.data,
        question_limit=args.question_limit,
        question_ids=load_question_ids(args.question_ids),
        question_types=parse_question_types(args.question_types),
    )
    return _apply_shard(samples, args.num_shards, args.shard_index)


def _apply_shard(samples: list[LongMemEvalSample], num_shards: int, shard_index: int) -> list[LongMemEvalSample]:
    if num_shards <= 1:
        return samples
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")
    return [sample for idx, sample in enumerate(samples) if idx % num_shards == shard_index]


def _top_k_cutoffs(args: argparse.Namespace) -> list[int]:
    if not args.top_k_list:
        return [args.top_k]
    values = [int(part.strip()) for part in args.top_k_list.split(",") if part.strip()]
    return sorted(dict.fromkeys(values))


def _prepare_cutoff_run_dirs(run_dir: Path, cutoffs: list[int], max_evidence_tokens: int | None) -> None:
    retrieval_results = read_jsonl(run_dir / "retrieval_results.jsonl")
    build_stats = read_jsonl(run_dir / "build_stats.jsonl")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8")) if (run_dir / "config.json").exists() else {}
    token_stats = json.loads((run_dir / "token_stats.json").read_text(encoding="utf-8")) if (run_dir / "token_stats.json").exists() else None
    for cutoff in cutoffs:
        cutoff_dir = ensure_dir(run_dir / f"top_{cutoff}")
        write_jsonl(
            cutoff_dir / "retrieval_results.jsonl",
            (_slice_retrieval_result(row, cutoff, max_evidence_tokens) for row in retrieval_results),
        )
        if build_stats:
            write_jsonl(cutoff_dir / "build_stats.jsonl", build_stats)
        cutoff_config = dict(config)
        cutoff_config["top_k"] = cutoff
        cutoff_config["parent_run_dir"] = str(run_dir)
        write_json(cutoff_dir / "config.json", cutoff_config)
        if token_stats is not None and not (cutoff_dir / "token_stats.json").exists():
            write_json(cutoff_dir / "token_stats.json", token_stats)


def _slice_retrieval_result(row: dict[str, Any], cutoff: int, max_evidence_tokens: int | None) -> dict[str, Any]:
    sliced = copy.deepcopy(row)
    matched_turns = sliced.get("matched_turns", [])[:cutoff]
    matched_ids = {turn.get("stable_turn_id") for turn in matched_turns}
    evidence_windows = [
        window for window in sliced.get("evidence_windows", []) if window.get("matched_stable_turn_id") in matched_ids
    ]
    deduped_evidence = _dedupe_and_sort_windows(evidence_windows)
    formatted = format_evidence_for_answerer(
        deduped_evidence,
        sliced.get("question_date", ""),
        max_evidence_tokens,
        question_type=sliced.get("question_type"),
    )
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


def _retrieval_metrics_from_row(
    row: dict[str, Any],
    matched_turns: list[dict[str, Any]],
    deduped_evidence: list[dict[str, Any]],
    evidence_token_count: int,
) -> dict[str, Any]:
    skipped = str(row.get("question_id", "")).endswith("_abs") or not row.get("answer_session_ids")
    answer_session_ids = {id_key(session_id) for session_id in row.get("answer_session_ids", [])}
    matched_session_ids = {id_key(turn.get("session_id", "")) for turn in matched_turns}
    deduped_session_ids = {id_key(turn.get("session_id", "")) for turn in deduped_evidence}
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


def _iter_parallel(items: list[Any], fn: Any, workers: int) -> Any:
    if workers <= 1:
        for item in items:
            yield fn(item)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fn, item) for item in items]
        for future in as_completed(futures):
            yield future.result()


def write_final_metrics(run_dir: Path, samples: list[LongMemEvalSample]) -> None:
    retrieval_results = read_jsonl(run_dir / "retrieval_results.jsonl")
    retrieval_metrics, retrieval_by_type = aggregate_retrieval_results(retrieval_results)
    write_json(run_dir / "retrieval_metrics.json", retrieval_metrics)
    write_json(run_dir / "retrieval_metrics_by_type.json", retrieval_by_type)

    judge_logs = read_jsonl(run_dir / "judge_logs.jsonl")
    if not judge_logs:
        write_json(run_dir / "metrics.json", retrieval_metrics)
        write_json(run_dir / "metrics_by_type.json", retrieval_by_type)
        return

    qa_metrics = _metrics(judge_logs)
    token_stats = json.loads((run_dir / "token_stats.json").read_text(encoding="utf-8")) if (run_dir / "token_stats.json").exists() else new_token_summary()
    write_json(run_dir / "qa_metrics.json", qa_metrics)
    write_json(run_dir / "metrics.json", {"retrieval": retrieval_metrics, "retrieval_by_type": retrieval_by_type, "qa": qa_metrics})
    write_error_analysis(
        samples,
        retrieval_results,
        run_dir / "predictions.jsonl",
        token_stats,
        run_dir / "error_analysis.jsonl",
        autoeval_log=run_dir / "judge_logs.jsonl",
    )


def _formatted_evidence_for_answer(
    result: dict[str, Any],
    sample: LongMemEvalSample,
    max_evidence_tokens: int | None,
    pruner: ProvenceEvidencePruner | None = None,
) -> dict[str, Any]:
    pruning = ProvencePruningResult([], 0, 0, 0, 0, 0.0, 0.0)
    if result.get("deduped_evidence"):
        evidence_turns = result["deduped_evidence"]
        if pruner is not None:
            pruning = pruner.prune_turns(sample.question, evidence_turns)
            evidence_turns = pruning.turns
        formatted = format_evidence_for_answerer(
            evidence_turns,
            sample.question_date,
            max_evidence_tokens,
            question_type=sample.question_type,
        )
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


def _load_or_new_token_stats(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "token_stats.json"
    if path.exists():
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    return new_token_summary()


def _reset_build_tokens(token_stats: dict[str, Any]) -> None:
    token_stats["build_tokens"] = {
        "embedding_input_tokens": 0,
        "embedding_provider_tokens": 0,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_total_tokens": 0,
    }
    token_stats.setdefault("time_stats", {})["build_seconds"] = 0.0
    token_stats["method_cost_tokens"]["build_embedding_input_tokens"] = 0
    for per_question in token_stats.get("per_question", {}).values():
        per_question.pop("build_tokens", None)
        per_question.pop("build_time_seconds", None)


def _reset_retrieval_embedding_tokens(token_stats: dict[str, Any]) -> None:
    token_stats["retrieval_embedding_tokens"] = {"input_tokens": 0}
    for per_question in token_stats.get("per_question", {}).values():
        per_question.pop("retrieval_embedding_tokens", None)


def _reset_query_tokens(token_stats: dict[str, Any]) -> None:
    token_stats["query_tokens"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    token_stats.setdefault("time_stats", {})["query_seconds"] = 0.0
    token_stats["method_cost_tokens"]["query_total_tokens"] = 0
    for per_question in token_stats.get("per_question", {}).values():
        per_question.pop("query_tokens", None)
        per_question.pop("query_input_tokens", None)
        per_question.pop("query_output_tokens", None)
        per_question.pop("query_time_seconds", None)


def _reset_judge_tokens(token_stats: dict[str, Any]) -> None:
    token_stats["judge_tokens"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    token_stats.setdefault("time_stats", {})["judge_seconds_excluded"] = 0.0
    token_stats["method_cost_tokens"]["judge_total_tokens_excluded"] = 0
    for per_question in token_stats.get("per_question", {}).values():
        per_question.pop("judge_tokens", None)
        per_question.pop("judge_time_seconds", None)


def _check_store(store_dir: Path, model_name: str, chunk_mode: str, embedding_text_mode: str) -> None:
    import json

    stats_path = store_dir / "store_stats.json"
    if not stats_path.exists():
        return
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    built_with = stats.get("embedder_model")
    if built_with and built_with != model_name:
        raise ValueError(
            f"{store_dir} was built with embedding model {built_with!r}, "
            f"but retrieve is using {model_name!r}. Re-run build with --force-rebuild."
        )
    built_chunk_mode = stats.get("chunk_mode", "turn")
    if built_chunk_mode != chunk_mode:
        raise ValueError(
            f"{store_dir} was built with chunk_mode {built_chunk_mode!r}, "
            f"but retrieve is using {chunk_mode!r}. Re-run build with --force-rebuild."
        )
    built_embedding_text_mode = stats.get("embedding_text_mode", "metadata_content")
    if built_embedding_text_mode != embedding_text_mode:
        raise ValueError(
            f"{store_dir} was built with embedding_text_mode {built_embedding_text_mode!r}, "
            f"but retrieve is using {embedding_text_mode!r}. Re-run build with --force-rebuild."
        )


def _config_dict(args: argparse.Namespace, samples: list[LongMemEvalSample]) -> dict[str, Any]:
    return {
        "data": args.data,
        "run_id": args.run_id,
        "top_k": args.top_k,
        "top_k_list": args.top_k_list,
        "message_range": args.message_range,
        "chunk_mode": args.chunk_mode,
        "embedding_text_mode": args.embedding_text_mode,
        "retrieval_method": args.retrieval_method,
        "temporal_boost": args.temporal_boost,
        "embedding_model": args.embedding_model,
        "embedding_backend": args.embedding_backend,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "question_limit": args.question_limit,
        "question_ids": args.question_ids,
        "question_types": args.question_types,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "parallelism": args.parallelism,
        "answer_parallelism": args.answer_parallelism,
        "judge_parallelism": args.judge_parallelism,
        "cutoff_parallelism": args.cutoff_parallelism,
        "mode": args.mode,
        "max_evidence_tokens": args.max_evidence_tokens,
        "disable_evidence_compiler": args.disable_evidence_compiler,
        "provence_pruning": args.provence_pruning,
        "provence_model": args.provence_model,
        "provence_threshold": args.provence_threshold,
        "provence_batch_size": args.provence_batch_size,
        "typed_sidecar": args.typed_sidecar,
        "output_dir": args.output_dir,
        "num_selected_samples": len(samples),
        "selected_question_ids": [sample.question_id for sample in samples],
    }


def _backend_arg(value: str | None) -> str | None:
    return None if value in {None, "auto"} else value


def main(argv: list[str] | None = None) -> None:
    result = run_pipeline(parse_args(argv))
    print(f"run_dir={result['run_dir']} num_samples={result['num_samples']}")


if __name__ == "__main__":
    main()
