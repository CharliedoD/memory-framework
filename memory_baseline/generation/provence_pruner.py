from __future__ import annotations

import time
from contextlib import redirect_stderr
from dataclasses import dataclass
from io import StringIO
from typing import Any

from memory_baseline.core.utils import estimate_tokens


DEFAULT_PROVENCE_MODEL = "naver/provence-reranker-debertav3-v1"


@dataclass(frozen=True)
class ProvencePruningResult:
    turns: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    source_turn_count: int
    kept_turn_count: int
    compression_rate: float
    latency_seconds: float


class ProvenceEvidencePruner:
    def __init__(
        self,
        model_name: str = DEFAULT_PROVENCE_MODEL,
        threshold: float = 0.1,
        batch_size: int = 32,
        model: Any | None = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            if torch.cuda.is_available():
                self._model.to("cuda")
            self._model.eval()
        return self._model

    def prune_turns(self, question: str, turns: list[dict[str, Any]]) -> ProvencePruningResult:
        started = time.monotonic()
        source_turns = [dict(turn) for turn in turns]
        if not source_turns:
            return ProvencePruningResult([], 0, 0, 0, 0, 0.0, time.monotonic() - started)

        non_empty_turns = [turn for turn in source_turns if str(turn.get("content", "")).strip()]
        contexts = [str(turn.get("content", "")) for turn in non_empty_turns]
        if not contexts:
            return ProvencePruningResult([], 0, 0, len(source_turns), 0, 0.0, time.monotonic() - started)
        with redirect_stderr(StringIO()):
            output = self.model.process(
                question,
                [contexts],
                title=None,
                batch_size=self.batch_size,
                threshold=self.threshold,
                always_select_title=False,
                enable_warnings=False,
        )
        pruned_contexts = output["pruned_context"][0]
        pruned_turns = []
        for turn, content in zip(non_empty_turns, pruned_contexts):
            pruned_content = str(content).strip()
            if not pruned_content:
                continue
            turn["content"] = pruned_content
            pruned_turns.append(turn)

        input_tokens = estimate_tokens("\n".join(contexts))
        output_tokens = estimate_tokens("\n".join(turn["content"] for turn in pruned_turns))
        compression_rate = 0.0 if input_tokens == 0 else (input_tokens - output_tokens) / input_tokens
        return ProvencePruningResult(
            turns=pruned_turns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            source_turn_count=len(source_turns),
            kept_turn_count=len(pruned_turns),
            compression_rate=compression_rate,
            latency_seconds=time.monotonic() - started,
        )


def make_provence_pruner(
    model_name: str = DEFAULT_PROVENCE_MODEL,
    threshold: float = 0.1,
    batch_size: int = 32,
) -> ProvenceEvidencePruner:
    return ProvenceEvidencePruner(model_name=model_name, threshold=threshold, batch_size=batch_size)
