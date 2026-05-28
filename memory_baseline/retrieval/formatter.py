from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory_baseline.core.utils import date_key, estimate_tokens, time_label, timestamp_sort_key


@dataclass(frozen=True)
class FormattedEvidence:
    text: str
    token_count: int
    truncated: bool
    included_turn_ids: list[str]
    truncated_turn_ids: list[str]
    truncate_strategy: str | None = None


def _sort_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        turns,
        key=lambda turn: (
            timestamp_sort_key(turn.get("session_date")),
            int(turn.get("session_idx", 0)),
            int(turn.get("turn_idx", 0)),
        ),
    )


def _render(
    turns: list[dict[str, Any]],
    question_date: str,
    question_type: str | None = None,
) -> str:
    lines = []
    if question_type == "temporal-reasoning":
        lines.extend(_render_temporal_timeline(turns, question_date))
        lines.append("")
    elif question_type == "multi-session":
        lines.extend(_render_count_and_list_check())
        lines.append("")

    lines.extend([
        "<RECALLED_MEMORY>",
        "These are recalled historical conversations from previous, separate sessions.",
        "The dates below are original benchmark timestamps, not the current runtime.",
        f"The user's question date is: {question_date}.",
        'When the user asks about "today", "now", "current", or relative time, interpret it relative to the question date.',
        "",
    ])
    current_date = None
    current_session = None
    for turn in _sort_turns(turns):
        group_date = date_key(turn.get("session_date"))
        if group_date != current_date:
            if current_date is not None:
                lines.append("")
            lines.append(f"## {group_date}")
            current_date = group_date
            current_session = None
        session_id = turn.get("session_id")
        if session_id != current_session:
            lines.append(
                f"### Session {session_id} | original timestamp: {turn.get('session_date')} | previous separate conversation"
            )
            current_session = session_id
        label = time_label(turn.get("turn_timestamp"), turn.get("session_date"))
        lines.append(f"[turn {int(turn.get('turn_idx', 0)):02d} | {turn.get('role')} | {label}] {turn.get('content')}")
    lines.append("</RECALLED_MEMORY>")
    return "\n".join(lines)


def _render_temporal_timeline(turns: list[dict[str, Any]], question_date: str) -> list[str]:
    lines = [
        "<TEMPORAL_TIMELINE>",
        "Candidate recalled turns sorted by original benchmark timestamp. This is only an index; use the raw recalled memory below as evidence.",
        f"Question date: {question_date}",
    ]
    for turn in _sort_turns(turns):
        lines.append(
            f"{turn.get('session_date')} | session {turn.get('session_id')} | turn {int(turn.get('turn_idx', 0)):02d} | {turn.get('role')} | {turn.get('stable_turn_id')} | {_clip(turn.get('content', ''))}"
        )
    lines.append("</TEMPORAL_TIMELINE>")
    return lines


def _render_count_and_list_check() -> list[str]:
    return [
        "<COUNT_AND_LIST_CHECK>",
        "Before giving the final answer, identify every distinct recalled item, amount, event, or session relevant to the question.",
        "Deduplicate repeated mentions, then compute the final count, list, or sum from those distinct items.",
        "Do not answer from only the first matching memory when several recalled sessions are relevant.",
        "</COUNT_AND_LIST_CHECK>",
    ]


def _clip(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def format_evidence_for_answerer(
    turns: list[dict[str, Any]],
    question_date: str,
    max_evidence_tokens: int | None = None,
    question_type: str | None = None,
) -> FormattedEvidence:
    sorted_turns = _sort_turns(turns)
    text = _render(sorted_turns, question_date, question_type)
    token_count = estimate_tokens(text)
    if max_evidence_tokens is None or token_count <= max_evidence_tokens:
        return FormattedEvidence(
            text=text,
            token_count=token_count,
            truncated=False,
            included_turn_ids=[turn["stable_turn_id"] for turn in sorted_turns],
            truncated_turn_ids=[],
        )

    kept: list[dict[str, Any]] = []
    for turn in sorted_turns:
        candidate = kept + [turn]
        if estimate_tokens(_render(candidate, question_date, question_type)) > max_evidence_tokens:
            break
        kept = candidate
    text = _render(kept, question_date, question_type)
    kept_ids = {turn["stable_turn_id"] for turn in kept}
    truncated_ids = [turn["stable_turn_id"] for turn in sorted_turns if turn["stable_turn_id"] not in kept_ids]
    return FormattedEvidence(
        text=text,
        token_count=estimate_tokens(text),
        truncated=True,
        included_turn_ids=[turn["stable_turn_id"] for turn in kept],
        truncated_turn_ids=truncated_ids,
        truncate_strategy="drop_tail_after_date_session_turn_sort",
    )
