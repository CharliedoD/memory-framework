from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

from memory_baseline.core.env import load_project_env
from memory_baseline.core.llm_cache import cached_response, write_cached_response
from memory_baseline.core.utils import estimate_messages_tokens, estimate_tokens
from memory_baseline.generation.judge import _open_json_with_retries


SYSTEM_PROMPT = """You compile retrieved long-term memory evidence for a downstream answerer.
Use only the retrieved evidence. Do not answer the question.
Keep source turn ids, dates, and short raw quotes for grounding.
If information is missing or conflicting, make that explicit."""


TYPE_INSTRUCTIONS = {
    "multi-session": "Extract every distinct relevant item, event, amount, entity, or session. Deduplicate repeated mentions and keep enough fields to support counts, lists, sums, and comparisons.",
    "knowledge-update": "Extract candidate values for the same subject or relation as a dated timeline. Mark older and newer conflicting values when the evidence supports that.",
    "temporal-reasoning": "Extract a dated timeline. Identify start/end events, first/latest/before/after relations, and any date arithmetic needed by the question. For relative durations like 'a month ago' and 'three weeks ago', the larger duration happened earlier.",
    "single-session-preference": "Extract user-specific preferences, constraints, plans, and prior choices relevant to the current request.",
}


@dataclass(frozen=True)
class EvidenceCompilerResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    provider_usage: dict[str, Any]
    latency_seconds: float
    model: str


class OpenAICompatibleEvidenceCompiler:
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def compile(
        self,
        question_date: str,
        evidence_text: str,
        question: str,
        question_type: str | None = None,
    ) -> EvidenceCompilerResult:
        messages = build_evidence_compiler_messages(question_date, evidence_text, question, question_type)
        payload_obj = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
        }
        cached = cached_response("evidence_compiler", self.model_name, payload_obj)
        if cached is not None:
            return _compiler_result_from_body(cached, messages, self.model_name, 0.0, cache_hit=True)
        payload = json.dumps(payload_obj).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.time()
        body = _open_json_with_retries(request)
        latency = time.time() - started
        write_cached_response("evidence_compiler", self.model_name, payload_obj, body)
        return _compiler_result_from_body(body, messages, self.model_name, latency, cache_hit=False)


def build_evidence_compiler_messages(
    question_date: str,
    evidence_text: str,
    question: str,
    question_type: str | None = None,
) -> list[dict[str, str]]:
    type_instruction = TYPE_INSTRUCTIONS.get(question_type or "", "Extract only the evidence relevant to answering the question.")
    user = f"""Question type: {question_type or "unknown"}
Question date: {question_date}
Question: {question}

Task-specific instruction:
{type_instruction}

For comparison questions, list the required comparison slots and mark any slot missing.
For unanswerable questions, mark which required information is missing.

Return concise Markdown in this exact wrapper:
<COMPILED_EVIDENCE>
Evidence status: sufficient|partial|insufficient
Required slots:
- ...
Structured facts:
| date | source_turn_id | fact/value | short quote |
| --- | --- | --- | --- |
...
Ordering or aggregation notes:
- ...
Missing or conflicting evidence:
- ...
</COMPILED_EVIDENCE>

Retrieved evidence:
{evidence_text}"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def make_evidence_compiler(
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAICompatibleEvidenceCompiler:
    load_project_env()
    model = model_name or os.getenv("ANSWER_MODEL") or os.getenv("ANSWERER_MODEL") or os.getenv("LLM_MODEL")
    key = api_key or os.getenv("ANSWER_API_KEY") or os.getenv("ANSWERER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not model:
        raise RuntimeError("ANSWER_MODEL or ANSWERER_MODEL is required for evidence compiler.")
    if not key:
        raise RuntimeError("ANSWER_API_KEY or ANSWERER_API_KEY is required for evidence compiler.")
    url = base_url or os.getenv("ANSWER_BASE_URL") or os.getenv("ANSWERER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return OpenAICompatibleEvidenceCompiler(model_name=model, base_url=url, api_key=key)


def _compiler_result_from_body(
    body: dict[str, Any],
    messages: list[dict[str, str]],
    model_name: str,
    latency_seconds: float,
    cache_hit: bool,
) -> EvidenceCompilerResult:
    compiled = body["choices"][0].get("message", {}).get("content", "").strip()
    usage = dict(body.get("usage", {}))
    prompt_tokens = int(usage.get("prompt_tokens") or estimate_messages_tokens(messages))
    completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(compiled))
    usage["cache_hit"] = cache_hit
    return EvidenceCompilerResult(
        text=compiled,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
        provider_usage=usage,
        latency_seconds=latency_seconds,
        model=model_name,
    )
