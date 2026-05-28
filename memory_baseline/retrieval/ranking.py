from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np

from memory_baseline.core.utils import timestamp_sort_key


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "with",
}
QUERY_KEYWORD_STOPWORDS = STOPWORDS | {
    "about",
    "also",
    "answer",
    "anything",
    "ask",
    "asked",
    "asking",
    "chat",
    "conversation",
    "conversations",
    "current",
    "currently",
    "day",
    "earlier",
    "earliest",
    "first",
    "latest",
    "last",
    "later",
    "memory",
    "mention",
    "mentioned",
    "now",
    "recent",
    "recently",
    "remember",
    "said",
    "say",
    "session",
    "sessions",
    "tell",
    "told",
    "today",
    "user",
    "yesterday",
    "you",
    "your",
}
SPEAKER_QUERY_STOPWORDS = {
    "can",
    "could",
    "he",
    "her",
    "him",
    "his",
    "many",
    "much",
    "she",
    "should",
    "that",
    "their",
    "them",
    "they",
    "us",
    "we",
    "would",
}


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = [_tokenize(text) for text in texts]
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        doc_freq: Counter[str] = Counter()
        for doc in self.documents:
            doc_freq.update(set(doc))
        total = len(self.documents)
        self.idf = {term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in doc_freq.items()}
        self.term_freqs = [Counter(doc) for doc in self.documents]

    def scores(self, query: str, extra_stopwords: set[str] | None = None) -> np.ndarray:
        query_terms = keyword_query_terms(query, extra_stopwords)
        scores = np.zeros(len(self.documents), dtype=np.float32)
        if not query_terms or not self.documents or self.avgdl == 0:
            return scores
        for idx, term_freq in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[idx]
            score = 0.0
            for term in query_terms:
                freq = term_freq.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1) / denom)
            scores[idx] = score
        return scores


def rank_indices(
    *,
    query: str,
    dense_scores: np.ndarray,
    raw_turns: list[dict[str, Any]],
    retrieval_texts: list[str],
    candidate_indices: list[int],
    top_k: int,
    retrieval_method: str = "dense",
    question_type: str | None = None,
    question_date: str | None = None,
    temporal_boost: float = 0.0,
    store: Any = None,
    ranking_diagnostics: dict[str, Any] | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray | None]:
    candidate_scores = dense_scores[candidate_indices].astype(np.float32, copy=True)
    bm25_scores = None
    speaker_aware = any(turn.get("speaker") for turn in raw_turns)
    if retrieval_method == "hybrid":
        bm25_scores = _bm25_scores(store, retrieval_texts, query)
        candidate_scores = _rrf_scores(candidate_scores, bm25_scores[candidate_indices])
    elif retrieval_method == "semantic_bm25_boost":
        bm25_scores = _bm25_scores(
            store,
            _semantic_bm25_texts(raw_turns, retrieval_texts),
            query,
            "_semantic_bm25_index",
            SPEAKER_QUERY_STOPWORDS if speaker_aware else None,
        )
        candidate_scores = _semantic_bm25_boost_scores(
            query=query,
            dense_scores=dense_scores,
            bm25_scores=bm25_scores,
            raw_turns=raw_turns,
            candidate_indices=candidate_indices,
            top_k=top_k,
            question_type=question_type,
            speaker_aware=speaker_aware,
            diagnostics=ranking_diagnostics,
        )
    elif retrieval_method != "dense":
        raise ValueError(f"Unsupported retrieval_method: {retrieval_method}")

    apply_temporal_boost = temporal_boost and question_type in {"temporal-reasoning", "temporal"}
    if retrieval_method == "semantic_bm25_boost":
        apply_temporal_boost = bool(apply_temporal_boost and _has_current_time_cue(query))
    if apply_temporal_boost:
        candidate_scores = candidate_scores + temporal_boost * _temporal_scores(raw_turns, candidate_indices, question_date)
    if ranking_diagnostics is not None and retrieval_method == "semantic_bm25_boost":
        ranking_diagnostics["temporal_boost_applied"] = bool(apply_temporal_boost)

    top_n = min(top_k, len(candidate_indices))
    ranked_candidate_positions = np.argsort(-candidate_scores)[:top_n]
    indices = [candidate_indices[int(pos)] for pos in ranked_candidate_positions]
    score_by_index = np.zeros(len(raw_turns), dtype=np.float32)
    score_by_index[candidate_indices] = candidate_scores
    return indices, score_by_index, bm25_scores


def _tokenize(text: str) -> list[str]:
    return [
        term
        for match in TOKEN_RE.finditer(text)
        if (term := match.group(0).lower()) not in STOPWORDS
    ]


def keyword_query_terms(text: str, extra_stopwords: set[str] | None = None) -> list[str]:
    terms: list[str] = []
    for match in TOKEN_RE.finditer(text):
        term = match.group(0).lower()
        if term in QUERY_KEYWORD_STOPWORDS or (extra_stopwords is not None and term in extra_stopwords):
            continue
        if len(term) < 3 and not any(char.isdigit() for char in term):
            continue
        terms.append(term)
    return terms


def _semantic_bm25_texts(raw_turns: list[dict[str, Any]], retrieval_texts: list[str]) -> list[str]:
    texts: list[str] = []
    for turn, text in zip(raw_turns, retrieval_texts):
        speaker = str(turn.get("speaker", "")).strip()
        if speaker:
            text = re.sub(rf"^\s*{re.escape(speaker)}\s*:\s*", "", text)
        texts.append(text)
    return texts


def _bm25_scores(
    store: Any,
    retrieval_texts: list[str],
    query: str,
    cache_attr: str = "_bm25_index",
    extra_query_stopwords: set[str] | None = None,
) -> np.ndarray:
    if store is not None and getattr(store, cache_attr, None) is not None:
        return getattr(store, cache_attr).scores(query, extra_query_stopwords)
    index = BM25Index(retrieval_texts)
    if store is not None:
        setattr(store, cache_attr, index)
    return index.scores(query, extra_query_stopwords)


def _rrf_scores(scores1: np.ndarray, scores2: np.ndarray, k: int = 60) -> np.ndarray:
    fused = 1.0 / (k + _ranks_desc(scores1))
    positive = scores2 > 0
    if np.any(positive):
        fused[positive] += 1.0 / (k + _ranks_desc(scores2[positive]))
    return fused


def _semantic_bm25_boost_scores(
    *,
    query: str,
    dense_scores: np.ndarray,
    bm25_scores: np.ndarray,
    raw_turns: list[dict[str, Any]],
    candidate_indices: list[int],
    top_k: int,
    question_type: str | None,
    speaker_aware: bool,
    diagnostics: dict[str, Any] | None,
) -> np.ndarray:
    dense_candidate_scores = dense_scores[candidate_indices].astype(np.float32, copy=True)
    bm25_candidate_scores = bm25_scores[candidate_indices].astype(np.float32, copy=True)
    positive_bm25 = bm25_candidate_scores > 0
    if not np.any(positive_bm25):
        if diagnostics is not None:
            diagnostics.update(
                {
                    "dense_pool_size": _dense_pool_size(len(candidate_indices), top_k, speaker_aware),
                    "bm25_positive_count": 0,
                    "bm25_positive_ratio": 0.0,
                    "lexical_rescue_count": 0,
                    "bm25_weight": 0.0,
                }
            )
        return dense_candidate_scores

    positive_ratio = float(np.sum(positive_bm25)) / len(bm25_candidate_scores)
    dense_pool_size = _dense_pool_size(len(candidate_indices), top_k, speaker_aware)
    dense_pool_positions = set(int(pos) for pos in np.argsort(-dense_candidate_scores)[:dense_pool_size])
    rescue_positions = _lexical_rescue_positions(
        bm25_candidate_scores,
        dense_pool_positions,
        top_k,
        positive_ratio,
        speaker_aware,
    )
    eligible_positions = dense_pool_positions | rescue_positions

    final_scores = np.full(len(candidate_indices), -np.inf, dtype=np.float32)
    eligible = np.asarray([idx in eligible_positions for idx in range(len(candidate_indices))], dtype=bool)
    bm25_weight = _bm25_weight(query, positive_ratio, speaker_aware)
    final_scores[eligible] = (
        _normalize_dense_scores(dense_candidate_scores[eligible])
        + bm25_weight * _normalize_positive_scores(bm25_candidate_scores[eligible])
        + _metadata_boost_scores(query, raw_turns, candidate_indices, question_type, eligible)[eligible]
    )
    if diagnostics is not None:
        diagnostics.update(
            {
                "dense_pool_size": dense_pool_size,
                "bm25_positive_count": int(np.sum(positive_bm25)),
                "bm25_positive_ratio": positive_ratio,
                "lexical_rescue_count": len(rescue_positions),
                "bm25_weight": bm25_weight,
            }
        )
    return final_scores


def _dense_pool_size(num_candidates: int, top_k: int, speaker_aware: bool) -> int:
    if speaker_aware:
        return min(num_candidates, max(top_k * 4, 40))
    return min(num_candidates, max(top_k * 8, 80))


def _lexical_rescue_positions(
    bm25_scores: np.ndarray,
    dense_pool_positions: set[int],
    top_k: int,
    positive_ratio: float,
    speaker_aware: bool,
) -> set[int]:
    if speaker_aware and positive_ratio > 0.3:
        return set()
    rescue_limit = min(3, top_k // 4)
    if rescue_limit <= 0:
        return set()
    positive_positions = np.flatnonzero(bm25_scores > 0)
    if len(positive_positions) == 0:
        return set()
    max_score = float(np.max(bm25_scores[positive_positions]))
    top_percent_count = max(1, int(math.ceil(len(positive_positions) * 0.01)))
    top_positions = set(int(pos) for pos in positive_positions[np.argsort(-bm25_scores[positive_positions])[:top_percent_count]])
    strong_positions = {
        int(pos)
        for pos in positive_positions
        if int(pos) in top_positions or float(bm25_scores[int(pos)]) >= 0.5 * max_score
    }
    rescue_candidates = [pos for pos in strong_positions if pos not in dense_pool_positions]
    rescue_candidates.sort(key=lambda pos: float(bm25_scores[pos]), reverse=True)
    return set(rescue_candidates[:rescue_limit])


def _normalize_dense_scores(scores: np.ndarray) -> np.ndarray:
    return np.clip((scores.astype(np.float32, copy=False) + 1.0) / 2.0, 0.0, 1.0)


def _normalize_positive_scores(scores: np.ndarray) -> np.ndarray:
    normalized = np.zeros(len(scores), dtype=np.float32)
    positive = scores > 0
    if not np.any(positive):
        return normalized
    positive_scores = scores[positive]
    minimum = float(np.min(positive_scores))
    maximum = float(np.max(positive_scores))
    if maximum == minimum:
        normalized[positive] = 1.0
    else:
        normalized[positive] = (positive_scores - minimum) / (maximum - minimum)
    return normalized


def _bm25_weight(query: str, positive_ratio: float, speaker_aware: bool) -> float:
    terms = keyword_query_terms(query, SPEAKER_QUERY_STOPWORDS if speaker_aware else None)
    if not terms:
        return 0.1
    if _has_hard_anchor(query, terms):
        weight = 0.5
    else:
        weight = 0.25
    if speaker_aware and positive_ratio > 0.5:
        return min(weight, 0.1)
    if speaker_aware and positive_ratio > 0.3:
        return min(weight, 0.2)
    return weight


def _has_hard_anchor(query: str, terms: list[str]) -> bool:
    if any(any(char.isdigit() for char in term) for term in terms):
        return True
    if re.search(r"['\"][^'\"]+['\"]", query):
        return True
    words = re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", query)
    return any(word.lower() not in QUERY_KEYWORD_STOPWORDS for word in words)


def _metadata_boost_scores(
    query: str,
    raw_turns: list[dict[str, Any]],
    candidate_indices: list[int],
    question_type: str | None,
    eligible: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(len(candidate_indices), dtype=np.float32)
    query_lower = query.lower()
    for position, candidate_idx in enumerate(candidate_indices):
        if not eligible[position]:
            continue
        turn = raw_turns[candidate_idx]
        role = str(turn.get("role", "")).lower()
        if question_type == "single-session-user" and role == "user":
            scores[position] += 0.05
        elif question_type == "single-session-assistant" and role == "assistant":
            scores[position] += 0.05
        speaker = str(turn.get("speaker", "")).strip().lower()
        if speaker and len(speaker) >= 3 and speaker in query_lower:
            scores[position] += 0.05
    return scores


def _has_current_time_cue(query: str) -> bool:
    terms = {match.group(0).lower() for match in TOKEN_RE.finditer(query)}
    return bool(terms & {"current", "currently", "latest", "now", "recent", "recently", "today"})


def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float32)
    return ranks


def _temporal_scores(raw_turns: list[dict[str, Any]], candidate_indices: list[int], question_date: str | None) -> np.ndarray:
    dates = [_as_datetime(raw_turns[idx].get("session_date")) for idx in candidate_indices]
    question_dt = _as_datetime(question_date)
    valid_dates = [date for date in dates if date is not None and (question_dt is None or date <= question_dt)]
    if not valid_dates:
        return np.zeros(len(candidate_indices), dtype=np.float32)
    min_ts = min(date.timestamp() for date in valid_dates)
    max_ts = max(date.timestamp() for date in valid_dates)
    span = max(max_ts - min_ts, 1.0)
    scores = []
    for date in dates:
        if date is None or (question_dt is not None and date > question_dt):
            scores.append(0.0)
        else:
            scores.append((date.timestamp() - min_ts) / span)
    return np.asarray(scores, dtype=np.float32)


def _as_datetime(value: Any) -> datetime | None:
    key = timestamp_sort_key(str(value) if value is not None else None)
    if key[0] == 0 and isinstance(key[1], datetime):
        return key[1]
    return None
