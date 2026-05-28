from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[" or text[0] == "{":
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            return obj["data"]
        raise ValueError(f"Unsupported JSON dataset shape in {path}")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value))
    return value.strip("_") or "empty"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    pieces = re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE)
    return max(1, math.ceil(len(pieces) * 0.8))


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(m.get("role", "")) + estimate_tokens(m.get("content", "")) for m in messages)


def normalize_embedding_text(text: str) -> str:
    return " ".join(text.split())


def model_cache_dir_name(model_name: str) -> str:
    return sha256_text(model_name)[:16]


def date_key(timestamp: str | None) -> str:
    if not timestamp:
        return "unknown-date"
    value = str(timestamp).strip()
    if "T" in value:
        return value.split("T", 1)[0]
    if " " in value:
        first = value.split(" ", 1)[0]
        if re.match(r"\d{4}-\d{2}-\d{2}", first):
            return first
    match = re.search(r"\d{4}-\d{2}-\d{2}", value)
    return match.group(0) if match else value


def timestamp_sort_key(timestamp: str | None) -> tuple[int, Any]:
    if not timestamp:
        return (1, "")
    value = str(timestamp).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d (%a) %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
    ):
        try:
            candidate = value.rstrip("Z")[:19] if fmt == "%Y-%m-%dT%H:%M:%S" else value
            return (0, datetime.strptime(candidate, fmt))
        except ValueError:
            continue
    return (1, value)


def time_label(*timestamps: str | None) -> str:
    for timestamp in timestamps:
        if not timestamp:
            continue
        match = re.search(r"(\d{1,2}:\d{2})(?::\d{2})?", str(timestamp))
        if match:
            return match.group(1)
    return "time unknown"


def id_key(value: Any) -> str:
    return str(value)
