from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from memory_baseline.core.utils import estimate_tokens, timestamp_sort_key
from memory_baseline.retrieval.ranking import BM25Index


AMOUNT_RE = re.compile(r"(?:[$]\s*(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|bucks?))", re.I)
AGE_PATTERNS = [
    re.compile(r"\b(?:i am|i'm|i’m|i just turned|i turned|i'm turning|i’m turning)\s+(\d{1,3})\b", re.I),
    re.compile(r"\bmy\s+([a-z][a-z -]{1,40}?)'?s\s+(\d{1,3})(?:st|nd|rd|th)?\s+birthday\b", re.I),
    re.compile(r"\bmy\s+([a-z][a-z -]{1,40}?)\s+is\s+(\d{1,3})\b", re.I),
]
QUOTE_RE = re.compile(r'"([^"]{2,80})"|“([^”]{2,80})”')
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
GENERIC_ADVICE_RE = re.compile(r"\b(?:you should|you can|make sure|consider|remember to|it's important to|here are|tips for)\b", re.I)
EVENT_RE = re.compile(
    r"\b(?:bought|ordered|spent|paid|cost|went|visited|moved|lived|camp(?:ed|ing)|read|created|made|played|watched|attended|started|joined|met|invited|recommended|liked|loved|prefer|favorite)\b",
    re.I,
)
SIDE_QUERY_STOPWORDS = {"called", "create", "created", "name", "named", "new", "update", "updated"}


def build_typed_facts(raw_turns: list[dict[str, Any]], max_quote_chars: int = 260) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for turn in raw_turns:
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        text = _strip_speaker_prefix(content, turn)
        pieces = _sentences(text)
        source_id = str(turn.get("stable_turn_id", ""))
        for piece_idx, sentence in enumerate(pieces[:4]):
            sentence = sentence.strip()
            if not _useful_sentence(sentence):
                continue
            facts.extend(_facts_from_sentence(turn, sentence, piece_idx, source_id, max_quote_chars))
    return _dedupe_facts(facts)


def select_typed_facts(question: str, facts: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    if not facts:
        return []
    projections = [str(fact.get("projection") or _fact_projection(fact)) for fact in facts]
    bm25 = BM25Index(projections).scores(question, SIDE_QUERY_STOPWORDS)
    type_scores = np.asarray([_type_score(question, fact) for fact in facts], dtype=np.float32)
    speaker_scores = np.asarray([_speaker_score(question, fact) for fact in facts], dtype=np.float32)
    scores = bm25 + type_scores + speaker_scores
    if float(np.max(scores)) <= 0:
        return []
    ranked = np.argsort(-scores)
    selected: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, str]] = set()
    for idx in ranked:
        if len(selected) >= limit:
            break
        if scores[int(idx)] <= 0:
            break
        fact = dict(facts[int(idx)])
        if fact.get("fact_type") in {"quoted_fact", "title_fact"} and not _title_query(question):
            continue
        fact["selection_score"] = float(scores[int(idx)])
        key = (str(fact.get("fact_type")), str(fact.get("source_quote")))
        if key in seen_sources:
            continue
        seen_sources.add(key)
        selected.append(fact)
    return selected


def render_typed_fact_pack(facts: list[dict[str, Any]], token_budget: int = 900) -> str:
    if not facts:
        return ""
    lines = [
        "<TYPED_FACT_MEMORY>",
        "Typed facts extracted from long-term memory. Use them as a compact index; use recalled source turns below to verify details.",
        "For list, count, comparison, or arithmetic questions, enumerate all relevant typed facts before deciding the final answer.",
        "",
        "<TYPED_FACTS>",
    ]
    kept = []
    for fact in facts:
        row = {key: value for key, value in fact.items() if key not in {"projection", "selection_score"}}
        candidate = kept + [f"- {json.dumps(row, ensure_ascii=False, sort_keys=True)}"]
        if estimate_tokens("\n".join(lines + candidate + ["</TYPED_FACTS>", "</TYPED_FACT_MEMORY>"])) > token_budget:
            break
        kept = candidate
    lines.extend(kept)
    lines.extend(["</TYPED_FACTS>", "</TYPED_FACT_MEMORY>"])
    return "\n".join(lines)


def prepend_typed_fact_pack(evidence_text: str, facts: list[dict[str, Any]]) -> str:
    pack = render_typed_fact_pack(facts)
    if not pack:
        return evidence_text
    return f"{pack}\n\n{evidence_text}"


def source_turn_ids(facts: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for fact in facts:
        for source_id in fact.get("source_turn_ids", []):
            ids.add(str(source_id))
    return ids


def should_use_typed_facts(question: str, question_type: str | None = None) -> bool:
    q = question.lower()
    if question_type == "temporal-reasoning":
        return True
    return bool(
        re.search(r"\b(?:money|spent|spend|cost|paid|expense|expenses|total|store|buy|bought|purchase)\b", q)
        or re.search(r"\b(?:age|older|younger|years old|grandma|grandfather|grandmother)\b", q)
        or re.search(r"\b(?:camped|camping|read|book|books|yoga)\b", q)
    )


def _facts_from_sentence(
    turn: dict[str, Any],
    sentence: str,
    piece_idx: int,
    source_id: str,
    max_quote_chars: int,
) -> list[dict[str, Any]]:
    facts = []
    quote = _clip(sentence, max_quote_chars)
    amounts = _amounts(sentence)
    for amount in amounts:
        facts.append(
            _fact(
                turn,
                source_id,
                piece_idx,
                "money_fact",
                quote,
                {
                    "amount_usd": amount,
                    "value": f"${amount:g}",
                    "relation": "money_amount",
                },
            )
        )
    for age_fact in _age_facts(sentence):
        facts.append(_fact(turn, source_id, piece_idx, "profile_fact", quote, age_fact))
    quoted = [part for match in QUOTE_RE.findall(sentence) for part in match if part]
    for title in quoted[:3]:
        facts.append(
            _fact(
                turn,
                source_id,
                piece_idx,
                "title_fact" if _looks_like_title_context(sentence) else "quoted_fact",
                quote,
                {"value": title, "relation": "quoted_title"},
            )
        )
    if _looks_like_media_fact(sentence):
        facts.append(_fact(turn, source_id, piece_idx, "media_fact", quote, {"relation": "image_or_media_context"}))
    if EVENT_RE.search(sentence) and _allow_event_fact(turn):
        facts.append(_fact(turn, source_id, piece_idx, _event_type(sentence), quote, {"relation": "event"}))
    return facts


def _fact(
    turn: dict[str, Any],
    source_id: str,
    piece_idx: int,
    fact_type: str,
    quote: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    speaker = str(turn.get("speaker") or "").strip()
    subject = speaker or ("user" if str(turn.get("role", "")).lower() == "user" else str(turn.get("role", "speaker")))
    fact = {
        "fact_id": f"{source_id}:fact{piece_idx}:{fact_type}:{len(quote)}",
        "fact_type": fact_type,
        "subject": fields.pop("subject", subject),
        "source_turn_ids": [source_id],
        "source_quote": quote,
        "session_date": turn.get("session_date"),
        "turn_timestamp": turn.get("turn_timestamp"),
        **fields,
    }
    fact["projection"] = _fact_projection(fact)
    return fact


def _fact_projection(fact: dict[str, Any]) -> str:
    values = [
        fact.get("fact_type"),
        fact.get("subject"),
        fact.get("relation"),
        fact.get("value"),
        fact.get("source_quote"),
    ]
    text = " ".join(str(value) for value in values if value)
    return f"{text} {_normalized_anchors(text)}".strip()


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = SENTENCE_RE.split(text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _strip_speaker_prefix(content: str, turn: dict[str, Any]) -> str:
    speaker = str(turn.get("speaker") or "").strip()
    if speaker:
        return re.sub(rf"^\s*{re.escape(speaker)}\s*:\s*", "", content).strip()
    return content


def _useful_sentence(sentence: str) -> bool:
    if len(sentence) < 12:
        return False
    if GENERIC_ADVICE_RE.search(sentence) and not AMOUNT_RE.search(sentence):
        return False
    return bool(AMOUNT_RE.search(sentence) or EVENT_RE.search(sentence) or QUOTE_RE.search(sentence) or any(pattern.search(sentence) for pattern in AGE_PATTERNS))


def _amounts(sentence: str) -> list[float]:
    amounts = []
    for match in AMOUNT_RE.finditer(sentence):
        value = match.group(1) or match.group(2)
        if value:
            amounts.append(float(value.replace(",", "")))
    return amounts


def _age_facts(sentence: str) -> list[dict[str, Any]]:
    facts = []
    for pattern in AGE_PATTERNS:
        for match in pattern.finditer(sentence):
            if len(match.groups()) == 1:
                facts.append({"subject": "user", "relation": "age", "value": int(match.group(1))})
            else:
                relation = match.group(1).strip().lower()
                facts.append({"subject": f"user's {relation}", "relation": "age", "value": int(match.group(2))})
    return facts


def _looks_like_title_context(sentence: str) -> bool:
    return bool(re.search(r"\b(?:book|read|playlist|song|movie|show|album|podcast)\b", sentence, re.I))


def _looks_like_media_fact(sentence: str) -> bool:
    return "[Image:" in sentence or bool(re.search(r"\b(?:caption:|query:)\b", sentence, re.I))


def _event_type(sentence: str) -> str:
    if AMOUNT_RE.search(sentence):
        return "money_fact"
    if re.search(r"\b(?:like|liked|love|loved|prefer|favorite)\b", sentence, re.I):
        return "preference_fact"
    if re.search(r"\b(?:read|book)\b", sentence, re.I):
        return "reading_fact"
    return "event_fact"


def _allow_event_fact(turn: dict[str, Any]) -> bool:
    if turn.get("speaker"):
        return True
    return str(turn.get("role", "")).lower() != "assistant"


def _type_score(question: str, fact: dict[str, Any]) -> float:
    q = question.lower()
    fact_type = str(fact.get("fact_type", ""))
    score = 0.0
    if fact_type == "money_fact" and re.search(r"\b(?:money|spent|spend|cost|paid|expense|expenses|store|buy|bought|purchase)\b", q):
        score += 1.0
    if fact_type == "profile_fact" and re.search(r"\b(?:age|older|younger|years old|grandma|mother|father|sister|brother|me)\b", q):
        score += 1.0
    if fact_type == "preference_fact" and re.search(r"\b(?:like|love|prefer|favorite|recommend)\b", q):
        score += 0.8
    return score


def _normalized_anchors(text: str) -> str:
    anchors = []
    lower = text.lower()
    if re.search(r"\bcamp(?:ed|ing|s)?\b", lower):
        anchors.append("camp camped camping")
    if re.search(r"\bread(?:ing)?\b", lower):
        anchors.append("read reading")
    if re.search(r"\bspent|spend|expense|cost|paid|bought|buy\b", lower):
        anchors.append("spent spend expense cost paid bought buy")
    if re.search(r"\byoga\b", lower):
        anchors.append("yoga")
    return " ".join(anchors)


def _speaker_score(question: str, fact: dict[str, Any]) -> float:
    subject = str(fact.get("subject", "")).strip().lower()
    if len(subject) >= 3 and subject in question.lower():
        return 0.4
    return 0.0


def _title_query(question: str) -> bool:
    return bool(re.search(r"\b(?:book|song|playlist|movie|show|podcast|title|called|name)\b", question, re.I))


def _dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped = []
    for fact in sorted(facts, key=lambda row: (timestamp_sort_key(row.get("session_date")), str(row.get("fact_id")))):
        key = (str(fact.get("fact_type")), str(fact.get("source_turn_ids")), str(fact.get("source_quote")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
