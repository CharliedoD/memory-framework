from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from memory_baseline.core.utils import ensure_dir, model_cache_dir_name, sha256_text, write_json


def cached_response(kind: str, model_name: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    path = _cache_path(kind, model_name, payload)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_cached_response(kind: str, model_name: str, payload: dict[str, Any], response: dict[str, Any]) -> None:
    path = _cache_path(kind, model_name, payload)
    tmp_path = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    write_json(tmp_path, response)
    os.replace(tmp_path, path)


def _cache_path(kind: str, model_name: str, payload: dict[str, Any]) -> Path:
    cache_root = Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))
    key = sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return ensure_dir(cache_root / kind / model_cache_dir_name(model_name)) / f"{key}.json"
