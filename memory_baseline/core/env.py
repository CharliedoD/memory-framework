from __future__ import annotations

import os
from pathlib import Path

_LOADED: set[Path] = set()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_env(path: str | Path | None = None, override: bool = False) -> Path | None:
    env_path = Path(path) if path else project_root() / ".env"
    if not env_path.exists():
        return None
    env_path = env_path.resolve()
    if env_path in _LOADED and not override:
        return env_path
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        key_value = _parse_env_line(raw_line)
        if key_value is None:
            continue
        key, value = key_value
        if override or key not in os.environ:
            os.environ[key] = value
    _LOADED.add(env_path)
    return env_path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = _strip_inline_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for idx, char in enumerate(value):
        if char in {"'", '"'}:
            quote = None if quote == char else char
        if char == "#" and quote is None and idx > 0 and value[idx - 1].isspace():
            return value[:idx].strip()
    return value
