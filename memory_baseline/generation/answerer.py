from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from memory_baseline.core.env import load_project_env
from memory_baseline.core.llm_cache import cached_response, write_cached_response
from memory_baseline.core.utils import estimate_messages_tokens, estimate_tokens


SYSTEM_PROMPT = """You are answering questions using recalled long-term memory.
Use only the recalled memory evidence and the current question.
The user's question date is: {question_date}.
Interpret relative dates such as today, yesterday, last week, current, now relative to the question date, not the runtime date.
If multiple memories conflict, prefer the memory with the latest relevant timestamp unless the question asks about an earlier time.
If the evidence is insufficient, answer that the information is not available.
Give a concise answer. Do not mention internal retrieval IDs unless necessary."""


ANSWER_FOCUS_BY_TYPE = {
    "multi-session": "Use all relevant recalled sessions. For counts, lists, or comparisons, combine evidence across sessions instead of answering from the first matching memory.",
    "temporal-reasoning": "Order the recalled memories by their original timestamps before answering. Compute relative dates against the user's question date.",
    "single-session-preference": "Extract the user's preference constraints from the recalled memory and tailor the answer to those constraints. Avoid generic suggestions when the memory gives a specific preference.",
}


@dataclass(frozen=True)
class AnswerResult:
    hypothesis: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    provider_usage: dict[str, Any]
    latency_seconds: float
    model: str


def answer_focus_for_question_type(question_type: str | None) -> str | None:
    return ANSWER_FOCUS_BY_TYPE.get(question_type or "")


def build_answer_messages(
    question_date: str,
    evidence_text: str,
    question: str,
    question_type: str | None = None,
) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT.format(question_date=question_date)
    focus = answer_focus_for_question_type(question_type)
    user = (
        f"{evidence_text}\n\n"
        "<QUESTION>\n"
        f"{question}\n"
        "</QUESTION>\n\n"
    )
    if focus:
        user += f"<ANSWER_FOCUS>\n{focus}\n</ANSWER_FOCUS>\n\n"
    user += "Expected answer format: concise natural language."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class OpenAICompatibleChatAnswerer:
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def answer(
        self,
        question_date: str,
        evidence_text: str,
        question: str,
        question_type: str | None = None,
    ) -> AnswerResult:
        messages = build_answer_messages(question_date, evidence_text, question, question_type)
        payload_obj = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
        }
        cached = cached_response("answer", self.model_name, payload_obj)
        if cached is not None:
            return _answer_result_from_body(cached, messages, self.model_name, 0.0, cache_hit=True)
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
        write_cached_response("answer", self.model_name, payload_obj, body)
        return _answer_result_from_body(body, messages, self.model_name, latency, cache_hit=False)


def make_answerer(
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAICompatibleChatAnswerer:
    load_project_env()
    model = model_name or os.getenv("ANSWER_MODEL") or os.getenv("ANSWERER_MODEL") or os.getenv("LLM_MODEL")
    key = api_key or os.getenv("ANSWER_API_KEY") or os.getenv("ANSWERER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not model:
        raise RuntimeError("ANSWER_MODEL or ANSWERER_MODEL is required for answer mode.")
    if not key:
        raise RuntimeError("ANSWER_API_KEY or ANSWERER_API_KEY is required for answer mode.")
    url = base_url or os.getenv("ANSWER_BASE_URL") or os.getenv("ANSWERER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return OpenAICompatibleChatAnswerer(model_name=model, base_url=url, api_key=key)


def _answer_result_from_body(
    body: dict[str, Any],
    messages: list[dict[str, str]],
    model_name: str,
    latency_seconds: float,
    cache_hit: bool,
) -> AnswerResult:
    choice = body["choices"][0]
    hypothesis = choice.get("message", {}).get("content", "").strip()
    usage = dict(body.get("usage", {}))
    prompt_tokens = int(usage.get("prompt_tokens") or estimate_messages_tokens(messages))
    completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(hypothesis))
    usage["cache_hit"] = cache_hit
    return AnswerResult(
        hypothesis=hypothesis,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
        provider_usage=usage,
        latency_seconds=latency_seconds,
        model=model_name,
    )


def _open_json_with_retries(request: urllib.request.Request) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
        except urllib.error.URLError as exc:
            last_error = exc
        except (TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(min(30, 5 * (attempt + 1)))
    assert last_error is not None
    raise last_error
