from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .embedder import BaseEmbedder, embed_texts_cached, l2_normalize
from memory_baseline.data.longmemeval import flatten_sample, save_normalized_question
from memory_baseline.core.schemas import LongMemEvalSample
from memory_baseline.core.utils import ensure_dir, read_jsonl, safe_filename, write_json, write_jsonl
from memory_baseline.retrieval.typed_facts import build_typed_facts


def embedding_text_for_turn(turn: dict[str, Any], embedding_text_mode: str = "content") -> str:
    if embedding_text_mode == "content":
        return str(turn.get("content", ""))
    if embedding_text_mode != "metadata_content":
        raise ValueError(f"Unsupported embedding_text_mode: {embedding_text_mode}")
    return (
        f"[date: {turn['session_date']}] "
        f"[session: {turn['session_id']}] "
        f"[role: {turn['role']}] "
        f"{turn['content']}"
    )


def embedding_texts_for_turns(
    raw_turns: list[dict[str, Any]],
    chunk_mode: str = "turn",
    embedding_text_mode: str = "content",
) -> list[str]:
    if chunk_mode != "turn":
        raise ValueError(f"Unsupported chunk_mode: {chunk_mode}")
    return [embedding_text_for_turn(turn, embedding_text_mode) for turn in raw_turns]


def lexical_texts_for_turns(raw_turns: list[dict[str, Any]], chunk_mode: str = "turn") -> list[str]:
    if chunk_mode != "turn":
        raise ValueError(f"Unsupported chunk_mode: {chunk_mode}")
    return [str(turn.get("content", "")) for turn in raw_turns]


def question_store_dir(run_dir: str | Path, question_id: str) -> Path:
    return Path(run_dir) / "stores" / safe_filename(question_id)


class QuestionStore:
    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.embeddings = l2_normalize(np.load(self.store_dir / "embeddings.npy"))
        self.raw_turns = read_jsonl(self.store_dir / "raw_turns.jsonl")
        self.embedding_meta = read_jsonl(self.store_dir / "embedding_meta.jsonl")
        self.stats = json.loads((self.store_dir / "store_stats.json").read_text(encoding="utf-8"))
        embedding_text_mode = self.stats.get("embedding_text_mode", "metadata_content")
        self.retrieval_texts = [
            row.get("embedding_text") or embedding_text_for_turn(turn, embedding_text_mode)
            for row, turn in zip(self.embedding_meta, self.raw_turns)
        ]
        self.lexical_texts = lexical_texts_for_turns(self.raw_turns, self.stats.get("chunk_mode", "turn"))
        typed_facts_path = self.store_dir / "typed_facts.jsonl"
        self.typed_facts = read_jsonl(typed_facts_path) if typed_facts_path.exists() else []


def build_question_store(
    sample: LongMemEvalSample,
    run_dir: str | Path,
    embedder: BaseEmbedder,
    cache_root: str | Path = ".cache/embeddings",
    force_rebuild: bool = False,
    skip_existing: bool = False,
    chunk_mode: str = "turn",
    embedding_text_mode: str = "content",
    typed_sidecar: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    store_dir = question_store_dir(run_dir, sample.question_id)
    stats_path = store_dir / "store_stats.json"
    if (store_dir / "embeddings.npy").exists() and not force_rebuild:
        if skip_existing and stats_path.exists():
            import json

            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            _check_store_mode(stats, chunk_mode, embedding_text_mode, store_dir)
            if typed_sidecar and not (store_dir / "typed_facts.jsonl").exists():
                _write_typed_facts_for_store(store_dir, stats)
            return stats
        if stats_path.exists():
            import json

            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            _check_store_mode(stats, chunk_mode, embedding_text_mode, store_dir)
            if typed_sidecar and not (store_dir / "typed_facts.jsonl").exists():
                _write_typed_facts_for_store(store_dir, stats)
            return stats
    if store_dir.exists() and force_rebuild:
        shutil.rmtree(store_dir)
    ensure_dir(store_dir)

    started = time.monotonic()
    turns = flatten_sample(sample)
    save_normalized_question(turns, run_dir / "normalized", sample.question_id)
    raw_turns = [turn.to_dict() for turn in turns]
    texts = embedding_texts_for_turns(raw_turns, chunk_mode, embedding_text_mode)
    batch = embed_texts_cached(embedder, texts, cache_root=cache_root, force=force_rebuild)

    ensure_dir(store_dir)
    np.save(store_dir / "embeddings.npy", batch.vectors)
    write_jsonl(store_dir / "raw_turns.jsonl", raw_turns)
    write_jsonl(
        store_dir / "embedding_meta.jsonl",
        (
            {
                **turn,
                "embedding_text": text,
            }
            for turn, text in zip(raw_turns, texts)
        ),
    )
    typed_fact_count = 0
    if typed_sidecar:
        typed_facts = build_typed_facts(raw_turns)
        write_jsonl(store_dir / "typed_facts.jsonl", typed_facts)
        typed_fact_count = len(typed_facts)
    stats = {
        "question_id": sample.question_id,
        "num_turns": len(raw_turns),
        "chunk_mode": chunk_mode,
        "embedding_text_mode": embedding_text_mode,
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


def _check_store_mode(stats: dict[str, Any], chunk_mode: str, embedding_text_mode: str, store_dir: Path) -> None:
    built_with = stats.get("chunk_mode", "turn")
    if built_with != chunk_mode:
        raise ValueError(f"{store_dir} was built with chunk_mode {built_with!r}; re-run build with --force-rebuild.")
    built_embedding_text_mode = stats.get("embedding_text_mode", "metadata_content")
    if built_embedding_text_mode != embedding_text_mode:
        raise ValueError(
            f"{store_dir} was built with embedding_text_mode {built_embedding_text_mode!r}; "
            f"re-run build with --force-rebuild."
        )


def _write_typed_facts_for_store(store_dir: Path, stats: dict[str, Any]) -> None:
    typed_facts = build_typed_facts(read_jsonl(store_dir / "raw_turns.jsonl"))
    write_jsonl(store_dir / "typed_facts.jsonl", typed_facts)
    stats["typed_sidecar"] = True
    stats["typed_fact_count"] = len(typed_facts)
    write_json(store_dir / "store_stats.json", stats)
