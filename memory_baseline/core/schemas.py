from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LongMemEvalSample:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_session_ids: list[Any]
    haystack_dates: list[str]
    haystack_sessions: list[list[dict[str, Any]]]
    answer_session_ids: list[Any]

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "LongMemEvalSample":
        return cls(
            question_id=str(record["question_id"]),
            question_type=str(record["question_type"]),
            question=str(record["question"]),
            answer=str(record.get("answer", "")),
            question_date=str(record.get("question_date", "")),
            haystack_session_ids=list(record.get("haystack_session_ids", [])),
            haystack_dates=list(record.get("haystack_dates", [])),
            haystack_sessions=list(record.get("haystack_sessions", [])),
            answer_session_ids=list(record.get("answer_session_ids", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawTurn:
    question_id: str
    question_type: str
    question_date: str
    session_idx: int
    session_id: Any
    session_date: str
    turn_idx: int
    global_turn_idx: int
    role: str
    content: str
    has_answer: bool
    answer_session_label: bool
    stable_turn_id: str
    turn_timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: Any
    input_tokens: int
    provider_tokens: int
    provider_usage: dict[str, Any]
    cache_hits: int = 0
    cache_misses: int = 0
