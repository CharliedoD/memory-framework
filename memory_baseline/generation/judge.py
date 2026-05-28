from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from memory_baseline.core.env import load_project_env
from memory_baseline.core.llm_cache import cached_response, write_cached_response
from memory_baseline.core.utils import estimate_messages_tokens, estimate_tokens


LOCOMO_JUDGE_SYSTEM_PROMPT = "You are evaluating conversational AI memory recall. Return JSON only with the format requested."

LOCOMO_JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


def get_longmemeval_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        if task == "temporal-reasoning":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        if task == "knowledge-update":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        if task == "single-session-preference":
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        raise NotImplementedError(task)
    template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    return template.format(question, answer, response)


@dataclass(frozen=True)
class JudgeResult:
    label: str
    score: float
    reasoning: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    provider_usage: dict[str, Any]
    latency_seconds: float
    model: str
    raw_response: str


class OpenAICompatibleJudge:
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def judge_locomo(self, question: str, answer: str, response: str) -> JudgeResult:
        messages = [
            {"role": "system", "content": LOCOMO_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": LOCOMO_JUDGE_PROMPT.format(question=question, gold_answer=answer, generated_answer=response)},
        ]
        payload_obj = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
        }
        cached = cached_response("judge", self.model_name, payload_obj)
        if cached is not None:
            return _locomo_judge_result_from_body(cached, messages, self.model_name, 0.0, cache_hit=True)
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
        write_cached_response("judge", self.model_name, payload_obj, body)
        return _locomo_judge_result_from_body(body, messages, self.model_name, latency, cache_hit=False)

    def judge_longmemeval(self, question_type: str, question: str, answer: str, response: str, abstention: bool = False) -> JudgeResult:
        messages = [{"role": "user", "content": get_longmemeval_prompt(question_type, question, answer, response, abstention=abstention)}]
        payload_obj = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 10,
        }
        cached = cached_response("judge", self.model_name, payload_obj)
        if cached is not None:
            return _longmemeval_judge_result_from_body(cached, messages, self.model_name, 0.0, cache_hit=True)
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
        write_cached_response("judge", self.model_name, payload_obj, body)
        return _longmemeval_judge_result_from_body(body, messages, self.model_name, latency, cache_hit=False)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def make_judge(
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAICompatibleJudge:
    load_project_env()
    model = model_name or os.getenv("JUDGE_MODEL")
    key = api_key or os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not model:
        raise RuntimeError("JUDGE_MODEL is required for judge mode.")
    if not key:
        raise RuntimeError("JUDGE_API_KEY is required for judge mode.")
    url = base_url or os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return OpenAICompatibleJudge(model_name=model, base_url=url, api_key=key)


def _locomo_judge_result_from_body(
    body: dict[str, Any],
    messages: list[dict[str, str]],
    model_name: str,
    latency_seconds: float,
    cache_hit: bool,
) -> JudgeResult:
    content = body["choices"][0].get("message", {}).get("content", "").strip()
    parsed = _parse_json_object(content)
    label = str(parsed.get("label", "")).strip().upper()
    if label not in {"CORRECT", "WRONG"}:
        label = "CORRECT" if re.search(r"\bCORRECT\b", content, flags=re.I) else "WRONG"
    usage = dict(body.get("usage", {}))
    prompt_tokens = int(usage.get("prompt_tokens") or estimate_messages_tokens(messages))
    completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(content))
    usage["cache_hit"] = cache_hit
    return JudgeResult(
        label=label,
        score=1.0 if label == "CORRECT" else 0.0,
        reasoning=str(parsed.get("reasoning", "")),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
        provider_usage=usage,
        latency_seconds=latency_seconds,
        model=model_name,
        raw_response=content,
    )


def _longmemeval_judge_result_from_body(
    body: dict[str, Any],
    messages: list[dict[str, str]],
    model_name: str,
    latency_seconds: float,
    cache_hit: bool,
) -> JudgeResult:
    content = body["choices"][0].get("message", {}).get("content", "").strip()
    usage = dict(body.get("usage", {}))
    prompt_tokens = int(usage.get("prompt_tokens") or estimate_messages_tokens(messages))
    completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(content))
    label = "CORRECT" if "yes" in content.lower() else "WRONG"
    usage["cache_hit"] = cache_hit
    return JudgeResult(
        label=label,
        score=1.0 if label == "CORRECT" else 0.0,
        reasoning="",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
        provider_usage=usage,
        latency_seconds=latency_seconds,
        model=model_name,
        raw_response=content,
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
