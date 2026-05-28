from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from memory_baseline.core.schemas import LongMemEvalSample


CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def load_locomo_records(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def sorted_session_items(conversation: dict[str, Any]) -> list[tuple[int, str, str, list[dict[str, Any]]]]:
    items = []
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", key)
        if match and isinstance(value, list):
            session_num = int(match.group(1))
            items.append((session_num, f"D{session_num}", conversation.get(f"{key}_date_time", ""), value))
    return sorted(items, key=lambda item: item[0])


def latest_session_date(record: dict[str, Any]) -> str:
    sessions = sorted_session_items(record["conversation"])
    return sessions[-1][2] if sessions else ""


def locomo_turn_content(turn: dict[str, Any]) -> str:
    text = str(turn.get("text", "")).strip()
    query = str(turn.get("query", "")).strip()
    caption = str(turn.get("blip_caption", "")).strip()
    image_bits = []
    if query:
        image_bits.append(f"query: {query}")
    if caption:
        image_bits.append(f"caption: {caption}")
    if image_bits:
        image_text = "[Image: " + "; ".join(image_bits) + "]"
        text = f"{text} {image_text}".strip()
    speaker = str(turn.get("speaker", "")).strip()
    return f"{speaker}: {text}" if speaker else text


def conversation_raw_turns(record: dict[str, Any], conv_idx: int) -> list[dict[str, Any]]:
    conversation = record["conversation"]
    speaker_a = conversation.get("speaker_a")
    turns = []
    global_turn_idx = 0
    for session_idx, (_session_num, session_id, session_date, session_turns) in enumerate(sorted_session_items(conversation)):
        for turn_idx, turn in enumerate(session_turns):
            dia_id = str(turn.get("dia_id") or f"{session_id}:{turn_idx}")
            turns.append(
                {
                    "question_id": f"conv{conv_idx}",
                    "question_type": "locomo-conversation",
                    "question_date": latest_session_date(record),
                    "session_idx": session_idx,
                    "session_id": session_id,
                    "session_date": session_date,
                    "turn_idx": turn_idx,
                    "global_turn_idx": global_turn_idx,
                    "role": "user" if turn.get("speaker") == speaker_a else "assistant",
                    "content": locomo_turn_content(turn),
                    "has_answer": False,
                    "answer_session_label": False,
                    "stable_turn_id": f"conv{conv_idx}:{dia_id}",
                    "turn_timestamp": session_date,
                    "locomo_dia_id": dia_id,
                    "speaker": turn.get("speaker", ""),
                }
            )
            global_turn_idx += 1
    return turns


def qa_answer(qa: dict[str, Any]) -> str:
    if "answer" in qa:
        return str(qa["answer"])
    return str(qa.get("adversarial_answer", "Not mentioned in the conversation"))


def qa_answer_sessions(qa: dict[str, Any]) -> list[str]:
    sessions = {str(evidence).split(":", 1)[0] for evidence in qa.get("evidence", []) if ":" in str(evidence)}
    return sorted(sessions, key=lambda value: (0, int(value[1:])) if value.startswith("D") and value[1:].isdigit() else (1, value))


def qa_sample(record: dict[str, Any], conv_idx: int, qa_idx: int, qa: dict[str, Any]) -> LongMemEvalSample:
    return LongMemEvalSample(
        question_id=f"conv{conv_idx}_q{qa_idx}",
        question_type=CATEGORY_NAMES.get(int(qa.get("category", 0)), "unknown"),
        question=str(qa["question"]),
        answer=qa_answer(qa),
        question_date=latest_session_date(record),
        haystack_session_ids=[],
        haystack_dates=[],
        haystack_sessions=[],
        answer_session_ids=qa_answer_sessions(qa),
    )


def iter_qa_items(
    records: list[dict[str, Any]],
    conversations: list[int],
    categories: list[int],
    question_limit: int | None = None,
) -> list[tuple[int, int, dict[str, Any], LongMemEvalSample]]:
    items = []
    for conv_idx in conversations:
        for qa_idx, qa in enumerate(records[conv_idx].get("qa", [])):
            if int(qa.get("category", 0)) not in categories:
                continue
            items.append((conv_idx, qa_idx, qa, qa_sample(records[conv_idx], conv_idx, qa_idx, qa)))
            if question_limit is not None and len(items) >= question_limit:
                return items
    return items
