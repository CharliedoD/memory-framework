from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from memory_baseline.core.env import load_project_env
from memory_baseline.core.schemas import EmbeddingBatch
from memory_baseline.core.utils import (
    ensure_dir,
    estimate_tokens,
    model_cache_dir_name,
    normalize_embedding_text,
    sha256_text,
)


class BaseEmbedder:
    model_name: str

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        raise NotImplementedError


class HashingEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "local-hash", dim: int = 384):
        self.model_name = model_name
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            if not tokens:
                vectors[row, 0] = 1.0
                continue
            for token in tokens:
                digest = int(sha256_text(token), 16)
                idx = digest % self.dim
                sign = 1.0 if ((digest >> 8) & 1) else -1.0
                vectors[row, idx] += sign
        vectors = l2_normalize(vectors)
        return EmbeddingBatch(
            vectors=vectors,
            input_tokens=sum(estimate_tokens(text) for text in texts),
            provider_tokens=0,
            provider_usage={},
        )


class OpenAICompatibleEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "16")))
        self.max_input_bytes = max(0, int(os.getenv("EMBEDDING_MAX_INPUT_BYTES", "8192")))

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(np.zeros((0, 0), dtype=np.float32), 0, 0, {})
        texts = [_truncate_utf8(text, self.max_input_bytes) for text in texts]
        started = time.time()
        vectors = []
        provider_tokens = 0
        provider_usage: dict[str, Any] = {"num_batches": 0}
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            body = self._embed_batch(batch_texts)
            provider_usage["num_batches"] += 1
            usage = body.get("usage", {})
            provider_tokens += int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
            _merge_usage(provider_usage, usage)
            data = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
            if len(data) != len(batch_texts):
                raise RuntimeError(f"Embedding API returned {len(data)} vectors for {len(batch_texts)} inputs.")
            vectors.extend(item["embedding"] for item in data)
        provider_usage["latency_seconds"] = time.time() - started
        vector_array = l2_normalize(np.asarray(vectors, dtype=np.float32))
        return EmbeddingBatch(
            vectors=vector_array,
            input_tokens=provider_tokens or sum(estimate_tokens(text) for text in texts),
            provider_tokens=provider_tokens,
            provider_usage=provider_usage,
        )

    def _embed_batch(self, texts: list[str]) -> dict[str, Any]:
        payload = json.dumps({"model": self.model_name, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embedding API error {exc.code}: {body}") from exc


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "4"))
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for local HF embeddings. "
                "Install it in this environment, or use --embedding-model local-hash for smoke tests."
            ) from exc
        self.model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs={"dtype": torch.bfloat16})

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(np.zeros((0, 0), dtype=np.float32), 0, 0, {})
        batch_size = self.batch_size
        while True:
            try:
                vectors = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
                break
            except RuntimeError as exc:
                if batch_size <= 1 or "out of memory" not in str(exc).lower():
                    raise
                next_batch_size = max(1, batch_size // 2)
                print(f"embedding OOM at batch_size={batch_size}; retrying with batch_size={next_batch_size}", flush=True)
                batch_size = next_batch_size
                release_cuda_cache()
        return EmbeddingBatch(
            vectors=l2_normalize(np.asarray(vectors, dtype=np.float32)),
            input_tokens=sum(estimate_tokens(text) for text in texts),
            provider_tokens=0,
            provider_usage={},
        )


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def release_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _merge_usage(target: dict[str, Any], usage: dict[str, Any]) -> None:
    for key, value in usage.items():
        if isinstance(value, int):
            target[key] = int(target.get(key, 0)) + value
        elif isinstance(value, float):
            target[key] = float(target.get(key, 0.0)) + value
        elif key not in target:
            target[key] = value


def _truncate_utf8(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return text
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")


def embedding_cache_key(model_name: str, text: str) -> str:
    normalized_text = normalize_embedding_text(text)
    return sha256_text(f"{model_name}\0{normalized_text}")


def embed_texts_cached(
    embedder: BaseEmbedder,
    texts: list[str],
    cache_root: str | Path = ".cache/embeddings",
    force: bool = False,
) -> EmbeddingBatch:
    cache_dir = ensure_dir(Path(cache_root) / model_cache_dir_name(embedder.model_name))
    texts = [_prepare_embedding_text(embedder, text) for text in texts]
    vectors: list[np.ndarray | None] = [None] * len(texts)
    misses: list[tuple[int, str, Path]] = []
    cache_hits = 0
    for idx, text in enumerate(texts):
        cache_path = cache_dir / f"{embedding_cache_key(embedder.model_name, text)}.npy"
        if cache_path.exists() and not force:
            try:
                vectors[idx] = np.load(cache_path)
                cache_hits += 1
            except Exception:
                misses.append((idx, text, cache_path))
        else:
            misses.append((idx, text, cache_path))

    provider_tokens = 0
    provider_usage: dict[str, Any] = {}
    if misses:
        batch = embedder.embed_texts([text for _, text, _ in misses])
        provider_tokens = batch.provider_tokens
        provider_usage = batch.provider_usage
        for row, (idx, _, cache_path) in enumerate(misses):
            vector = batch.vectors[row]
            _save_vector_atomic(cache_path, vector)
            vectors[idx] = vector

    stacked = np.vstack([vector for vector in vectors if vector is not None]) if vectors else np.zeros((0, 0))
    return EmbeddingBatch(
        vectors=l2_normalize(stacked),
        input_tokens=sum(estimate_tokens(text) for text in texts),
        provider_tokens=provider_tokens,
        provider_usage=provider_usage,
        cache_hits=cache_hits,
        cache_misses=len(misses),
    )


def _prepare_embedding_text(embedder: BaseEmbedder, text: str) -> str:
    max_input_bytes = int(getattr(embedder, "max_input_bytes", 0) or 0)
    return _truncate_utf8(text, max_input_bytes)


def _save_vector_atomic(cache_path: Path, vector: np.ndarray) -> None:
    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp_path.open("wb") as f:
        np.save(f, vector)
    os.replace(tmp_path, cache_path)


def make_embedder(
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    backend: str | None = None,
) -> BaseEmbedder:
    load_project_env()
    model = model_name or os.getenv("EMBEDDING_MODEL") or os.getenv("EMBEDDER_MODEL") or "local-hash"
    selected_backend = backend or os.getenv("EMBEDDING_BACKEND")
    if model in {"local-hash", "hash", "hashing"}:
        return HashingEmbedder(model_name=model)
    if selected_backend in {"hf", "sentence-transformers", "sentence_transformers"}:
        return SentenceTransformerEmbedder(model)
    key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("EMBEDDER_API_KEY")
    url = base_url or os.getenv("EMBEDDING_BASE_URL") or os.getenv("EMBEDDER_BASE_URL")
    if selected_backend in {"api", "openai-compatible", "openai_compatible"}:
        if not key:
            raise RuntimeError("EMBEDDING_API_KEY is required for OpenAI-compatible embedding mode.")
        return OpenAICompatibleEmbedder(model_name=model, base_url=url or "https://api.openai.com/v1", api_key=key)
    if not key and not url and ("/" in model or "qwen" in model.lower()):
        return SentenceTransformerEmbedder(model)
    if not key:
        raise RuntimeError(
            "EMBEDDING_API_KEY is required for non-local embedding models. "
            "Use --embedding-model local-hash for retrieval-only smoke tests without an API key."
        )
    return OpenAICompatibleEmbedder(model_name=model, base_url=url or "https://api.openai.com/v1", api_key=key)
