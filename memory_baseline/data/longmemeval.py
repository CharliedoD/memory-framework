from __future__ import annotations

from pathlib import Path
from typing import Iterable

from memory_baseline.core.schemas import LongMemEvalSample, RawTurn
from memory_baseline.core.utils import id_key, read_json_records, safe_filename, write_jsonl


def load_samples(
    data_path: str | Path,
    question_limit: int | None = None,
    question_ids: set[str] | None = None,
    question_types: set[str] | None = None,
) -> list[LongMemEvalSample]:
    records = read_json_records(data_path)
    samples: list[LongMemEvalSample] = []
    for record in records:
        sample = LongMemEvalSample.from_dict(record)
        if question_ids is not None and sample.question_id not in question_ids:
            continue
        if question_types is not None and sample.question_type not in question_types:
            continue
        samples.append(sample)
        if question_limit is not None and len(samples) >= question_limit:
            break
    return samples


def load_question_ids(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def parse_question_types(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def flatten_sample(sample: LongMemEvalSample) -> list[RawTurn]:
    if len(sample.haystack_session_ids) != len(sample.haystack_dates):
        raise ValueError(f"{sample.question_id}: haystack_session_ids and haystack_dates length mismatch")
    if len(sample.haystack_session_ids) != len(sample.haystack_sessions):
        raise ValueError(f"{sample.question_id}: haystack_session_ids and haystack_sessions length mismatch")

    answer_session_keys = {id_key(session_id) for session_id in sample.answer_session_ids}
    turns: list[RawTurn] = []
    global_turn_idx = 0
    for session_idx, (session_id, session_date, session) in enumerate(
        zip(sample.haystack_session_ids, sample.haystack_dates, sample.haystack_sessions)
    ):
        answer_session_label = id_key(session_id) in answer_session_keys
        for turn_idx, turn in enumerate(session):
            stable_turn_id = f"{sample.question_id}:{session_id}:{turn_idx}"
            turn_timestamp = (
                turn.get("timestamp")
                or turn.get("created_at")
                or turn.get("date")
                or turn.get("time")
            )
            turns.append(
                RawTurn(
                    question_id=sample.question_id,
                    question_type=sample.question_type,
                    question_date=sample.question_date,
                    session_idx=session_idx,
                    session_id=session_id,
                    session_date=str(session_date),
                    turn_idx=turn_idx,
                    global_turn_idx=global_turn_idx,
                    role=str(turn.get("role", "")),
                    content=str(turn.get("content", "")),
                    has_answer=bool(turn.get("has_answer", False)),
                    answer_session_label=answer_session_label,
                    stable_turn_id=stable_turn_id,
                    turn_timestamp=str(turn_timestamp) if turn_timestamp is not None else None,
                )
            )
            global_turn_idx += 1
    return turns


def save_normalized_question(turns: Iterable[RawTurn], output_dir: str | Path, question_id: str) -> Path:
    output_path = Path(output_dir) / f"{safe_filename(question_id)}.jsonl"
    write_jsonl(output_path, (turn.to_dict() for turn in turns))
    return output_path


def is_abstention_sample(sample: LongMemEvalSample | dict) -> bool:
    question_id = sample.question_id if isinstance(sample, LongMemEvalSample) else str(sample.get("question_id", ""))
    answer_session_ids = sample.answer_session_ids if isinstance(sample, LongMemEvalSample) else sample.get("answer_session_ids", [])
    return question_id.endswith("_abs") or not answer_session_ids
