from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np

from memory_baseline.indexing.embedder import BaseEmbedder, embed_texts_cached
from .formatter import format_evidence_for_answerer
from memory_baseline.data.longmemeval import is_abstention_sample
from memory_baseline.core.schemas import LongMemEvalSample
from memory_baseline.core.utils import estimate_tokens, id_key, timestamp_sort_key
from memory_baseline.indexing.vector_store import QuestionStore
from memory_baseline.retrieval.ranking import rank_indices
from memory_baseline.retrieval.typed_facts import prepend_typed_fact_pack, select_typed_facts, should_use_typed_facts, source_turn_ids


class DenseRetriever:
    def __init__(
        self,
        store: QuestionStore,
        embedder: BaseEmbedder,
        cache_root: str | Path = ".cache/embeddings",
    ):
        self.store = store
        self.embedder = embedder
        self.cache_root = cache_root

    def retrieve(
        self,
        sample: LongMemEvalSample,
        top_k: int,
        message_range: int,
        max_evidence_tokens: int | None = None,
        filters: dict[str, Any] | None = None,
        retrieval_method: str = "dense",
        temporal_boost: float = 0.0,
        query_vector: np.ndarray | None = None,
        query_embedding_tokens: int | None = None,
        typed_sidecar: bool = False,
    ) -> dict[str, Any]:
        started = time.monotonic()
        if query_vector is None:
            query_batch = embed_texts_cached(self.embedder, [sample.question], cache_root=self.cache_root)
            query = query_batch.vectors[0]
            query_tokens = query_batch.input_tokens
        else:
            query = query_vector
            query_tokens = int(query_embedding_tokens or 0)
        scores = self.store.embeddings @ query
        candidate_indices = _filter_indices(self.store.raw_turns, filters, sample.question_date)
        ranking_diagnostics: dict[str, Any] = {}
        indices, ranked_scores, _bm25_scores = rank_indices(
            query=sample.question,
            dense_scores=scores,
            raw_turns=self.store.raw_turns,
            retrieval_texts=self.store.lexical_texts,
            candidate_indices=candidate_indices,
            top_k=top_k,
            retrieval_method=retrieval_method,
            question_type=sample.question_type,
            question_date=sample.question_date,
            temporal_boost=temporal_boost,
            store=self.store,
            ranking_diagnostics=ranking_diagnostics,
        )
        matched_turns = []
        for rank, idx in enumerate(indices, start=1):
            turn = dict(self.store.raw_turns[int(idx)])
            matched_turns.append(_matched_turn(turn, float(ranked_scores[int(idx)]), rank))

        evidence_windows = _expand_windows(self.store.raw_turns, matched_turns, message_range)
        deduped_evidence = _dedupe_and_sort_windows(evidence_windows)
        typed_facts = []
        if typed_sidecar and should_use_typed_facts(sample.question, sample.question_type):
            evidence_ids = {str(turn.get("stable_turn_id")) for turn in deduped_evidence}
            candidate_facts = [
                fact
                for fact in self.store.typed_facts
                if evidence_ids & {str(source_id) for source_id in fact.get("source_turn_ids", [])}
            ]
            typed_facts = select_typed_facts(sample.question, candidate_facts)
        formatted = format_evidence_for_answerer(
            deduped_evidence,
            sample.question_date,
            max_evidence_tokens,
            question_type=sample.question_type,
        )
        formatted_text = prepend_typed_fact_pack(formatted.text, typed_facts)
        metrics = compute_retrieval_metrics(sample, matched_turns, deduped_evidence, estimate_tokens(formatted_text))

        result = {
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "question": sample.question,
            "answer": sample.answer,
            "question_date": sample.question_date,
            "answer_session_ids": sample.answer_session_ids,
            "top_k": top_k,
            "message_range": message_range,
            "retrieval_method": retrieval_method,
            "temporal_boost": temporal_boost,
            "timestamp_filter": {
                "question_date": sample.question_date,
                "num_candidates_before_filter": len(self.store.raw_turns),
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
            "query_embedding_tokens": query_tokens,
            "retrieval_latency_seconds": time.monotonic() - started,
            "metrics": metrics,
        }
        if ranking_diagnostics:
            result["ranking_diagnostics"] = ranking_diagnostics
        return result


def _matched_turn(turn: dict[str, Any], score: float, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "stable_turn_id": turn["stable_turn_id"],
        "score": score,
        "session_id": turn["session_id"],
        "session_date": turn["session_date"],
        "session_idx": turn["session_idx"],
        "turn_idx": turn["turn_idx"],
        "role": turn["role"],
        "content": turn["content"],
        "has_answer": bool(turn.get("has_answer", False)),
        "answer_session_label": bool(turn.get("answer_session_label", False)),
    }


def _filter_indices(
    raw_turns: list[dict[str, Any]],
    filters: dict[str, Any] | None,
    question_date: str | None = None,
) -> list[int]:
    question_dt = _as_datetime(question_date)
    return [
        idx
        for idx, turn in enumerate(raw_turns)
        if (not filters or all(turn.get(key) == value for key, value in filters.items()))
        and _is_at_or_before_question_date(turn.get("session_date"), question_dt)
    ]


def _is_at_or_before_question_date(session_date: Any, question_dt: datetime | None) -> bool:
    if question_dt is None:
        return True
    session_dt = _as_datetime(session_date)
    if session_dt is None:
        return True
    return session_dt <= question_dt or session_dt.date() == question_dt.date()


def _as_datetime(value: Any) -> datetime | None:
    key = timestamp_sort_key(str(value) if value is not None else None)
    if key[0] == 0 and isinstance(key[1], datetime):
        return key[1]
    return None


def _expand_windows(
    raw_turns: list[dict[str, Any]],
    matched_turns: list[dict[str, Any]],
    message_range: int,
) -> list[dict[str, Any]]:
    by_session: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for turn in raw_turns:
        key = (id_key(turn["session_id"]), int(turn["session_idx"]))
        by_session.setdefault(key, []).append(turn)
    for turns in by_session.values():
        turns.sort(key=lambda turn: int(turn["turn_idx"]))

    windows = []
    for match in matched_turns:
        key = (id_key(match["session_id"]), int(match["session_idx"]))
        turns = by_session[key]
        pos = next(idx for idx, turn in enumerate(turns) if turn["stable_turn_id"] == match["stable_turn_id"])
        start = max(0, pos - message_range)
        end = min(len(turns), pos + message_range + 1)
        windows.append(
            {
                "matched_stable_turn_id": match["stable_turn_id"],
                "start_turn_idx": turns[start]["turn_idx"],
                "end_turn_idx": turns[end - 1]["turn_idx"],
                "turns": [dict(turn) for turn in turns[start:end]],
            }
        )
    return windows


def _dedupe_and_sort_windows(evidence_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    turns: list[dict[str, Any]] = []
    for window in evidence_windows:
        for turn in window["turns"]:
            if turn["stable_turn_id"] in seen:
                continue
            seen.add(turn["stable_turn_id"])
            turns.append(dict(turn))
    return sorted(
        turns,
        key=lambda turn: (
            timestamp_sort_key(turn.get("session_date")),
            int(turn.get("session_idx", 0)),
            int(turn.get("turn_idx", 0)),
        ),
    )


def _dedupe_and_sort_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for turn in turns:
        turn_id = str(turn["stable_turn_id"])
        if turn_id in seen:
            continue
        seen.add(turn_id)
        deduped.append(dict(turn))
    return sorted(
        deduped,
        key=lambda turn: (
            timestamp_sort_key(turn.get("session_date")),
            int(turn.get("session_idx", 0)),
            int(turn.get("turn_idx", 0)),
        ),
    )


def _source_turns_for_typed_facts(raw_turns: list[dict[str, Any]], typed_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = source_turn_ids(typed_facts)
    if not wanted:
        return []
    return [dict(turn) for turn in raw_turns if str(turn.get("stable_turn_id")) in wanted]


def compute_retrieval_metrics(
    sample: LongMemEvalSample,
    matched_turns: list[dict[str, Any]],
    deduped_evidence: list[dict[str, Any]],
    evidence_token_count: int | None = None,
) -> dict[str, Any]:
    skipped = is_abstention_sample(sample)
    answer_session_keys = {id_key(session_id) for session_id in sample.answer_session_ids}
    matched_session_ids = {id_key(turn["session_id"]) for turn in matched_turns}
    deduped_session_ids = {id_key(turn["session_id"]) for turn in deduped_evidence}
    session_hit = bool(answer_session_keys & matched_session_ids) if not skipped else None
    turn_hit = any(turn.get("has_answer") for turn in matched_turns) if not skipped else None
    expanded_turn_hit = any(turn.get("has_answer") for turn in deduped_evidence) if not skipped else None
    return {
        "skipped": skipped,
        "session_recall_at_k": session_hit,
        "turn_recall_at_k": turn_hit,
        "expanded_turn_recall_at_k": expanded_turn_hit,
        "turn_or_expanded_recall_at_k": (bool(turn_hit or expanded_turn_hit) if not skipped else None),
        "evidence_token_count": evidence_token_count if evidence_token_count is not None else estimate_tokens(str(deduped_evidence)),
        "num_matched_turns": len(matched_turns),
        "num_deduped_turns": len(deduped_evidence),
        "num_sessions_recalled": len(answer_session_keys & matched_session_ids) if not skipped else 0,
        "num_deduped_sessions": len(deduped_session_ids),
    }
