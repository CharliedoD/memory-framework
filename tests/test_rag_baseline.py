from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from memory_baseline.generation.answerer import build_answer_messages
from memory_baseline.generation.evidence_compiler import build_evidence_compiler_messages
from memory_baseline.generation.judge import LOCOMO_JUDGE_PROMPT
from memory_baseline.generation.provence_pruner import ProvenceEvidencePruner
from memory_baseline.cli.run_locomo import _slice_retrieval_result as _slice_locomo_retrieval_result
from memory_baseline.cli.run_longmemeval import _apply_shard, _slice_retrieval_result as _slice_longmemeval_retrieval_result
from memory_baseline.cli.merge_longmemeval_metrics import _merge_config
from memory_baseline.core.llm_cache import cached_response, write_cached_response
from memory_baseline.retrieval.ranking import BM25Index, keyword_query_terms, rank_indices
from memory_baseline.retrieval.formatter import format_evidence_for_answerer
from memory_baseline.retrieval.typed_facts import build_typed_facts, prepend_typed_fact_pack, select_typed_facts, should_use_typed_facts, source_turn_ids
from memory_baseline.core.io import write_predictions
from memory_baseline.data.longmemeval import flatten_sample
from memory_baseline.evaluation.error_analysis import likely_failure_type, write_error_analysis
from memory_baseline.retrieval.dense import _dedupe_and_sort_windows, _expand_windows, _filter_indices, compute_retrieval_metrics
from memory_baseline.core.schemas import LongMemEvalSample
from memory_baseline.core.token_accounting import add_build_tokens, add_judge_tokens, add_query_tokens, new_token_summary
from memory_baseline.core.utils import write_json
from memory_baseline.indexing.vector_store import embedding_text_for_turn, lexical_texts_for_turns


def sample() -> LongMemEvalSample:
    return LongMemEvalSample.from_dict(
        {
            "question_id": "q1",
            "question_type": "temporal-reasoning",
            "question": "What did I say today?",
            "answer": "You said the blue notebook was in the desk.",
            "question_date": "2023-05-03",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": ["2023-04-17 09:20", "2023-05-02"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I bought a red pen."},
                    {"role": "assistant", "content": "Noted."},
                ],
                [
                    {"role": "user", "content": "The blue notebook is in the desk.", "has_answer": True},
                    {"role": "assistant", "content": "I will remember that."},
                ],
            ],
            "answer_session_ids": ["s2"],
        }
    )


def test_longmemeval_loader_preserves_dates():
    turns = flatten_sample(sample())
    assert turns[0].session_date == "2023-04-17 09:20"
    assert turns[2].session_date == "2023-05-02"
    assert turns[0].question_date == "2023-05-03"


def test_question_date_in_answer_prompt():
    messages = build_answer_messages("2023-05-03", "<RECALLED_MEMORY>x</RECALLED_MEMORY>", "What is today?")
    joined = "\n".join(message["content"] for message in messages)
    assert "2023-05-03" in joined
    assert "not the runtime date" in joined


def test_question_type_focus_in_answer_prompt():
    messages = build_answer_messages("2023-05-03", "<RECALLED_MEMORY>x</RECALLED_MEMORY>", "How many?", "multi-session")
    joined = "\n".join(message["content"] for message in messages)
    assert "<ANSWER_FOCUS>" in joined
    assert "combine evidence across sessions" in joined


def test_evidence_compiler_prompt_preserves_grounding():
    messages = build_evidence_compiler_messages(
        "2023-05-03",
        "<RECALLED_MEMORY>[turn 00 | user | 09:00] I bought tea.</RECALLED_MEMORY>",
        "How many teas did I buy?",
        "multi-session",
    )
    joined = "\n".join(message["content"] for message in messages)
    assert "Do not answer the question" in joined
    assert "source_turn_id" in joined
    assert "short raw quotes" in joined
    assert "Deduplicate repeated mentions" in joined


def test_has_answer_mapping():
    turns = flatten_sample(sample())
    assert [turn.has_answer for turn in turns] == [False, False, True, False]


def test_content_embedding_text_excludes_metadata():
    turn = flatten_sample(sample())[0].to_dict()
    text = embedding_text_for_turn(turn)
    assert text == turn["content"]
    assert "[date:" not in text
    assert "[session:" not in text
    assert "[role:" not in text


def test_answer_session_mapping():
    turns = flatten_sample(sample())
    assert [turn.answer_session_label for turn in turns] == [False, False, True, True]


def test_message_range_expansion():
    turns = [turn.to_dict() for turn in flatten_sample(sample())]
    match = {
        "stable_turn_id": turns[2]["stable_turn_id"],
        "session_id": "s2",
        "session_idx": 1,
    }
    windows = _expand_windows(turns, [match], message_range=1)
    assert [turn["turn_idx"] for turn in windows[0]["turns"]] == [0, 1]


def test_window_deduplication():
    turns = [turn.to_dict() for turn in flatten_sample(sample())]
    windows = [
        {"turns": [turns[2], turns[3]]},
        {"turns": [turns[3]]},
    ]
    deduped = _dedupe_and_sort_windows(windows)
    assert [turn["stable_turn_id"] for turn in deduped] == [turns[2]["stable_turn_id"], turns[3]["stable_turn_id"]]


def test_evidence_grouped_by_date():
    turns = [turn.to_dict() for turn in flatten_sample(sample())]
    formatted = format_evidence_for_answerer(turns, "2023-05-03")
    assert "## 2023-04-17" in formatted.text
    assert "## 2023-05-02" in formatted.text
    assert "### Session s2" in formatted.text
    assert "[turn 00 | user" in formatted.text


def test_temporal_evidence_includes_timeline():
    turns = [turn.to_dict() for turn in flatten_sample(sample())]
    formatted = format_evidence_for_answerer(turns, "2023-05-03", question_type="temporal-reasoning")
    assert "<TEMPORAL_TIMELINE>" in formatted.text
    assert "Question date: 2023-05-03" in formatted.text
    assert "q1:s2:0" in formatted.text
    assert "<RECALLED_MEMORY>" in formatted.text


def test_multi_session_evidence_includes_count_check():
    turns = [turn.to_dict() for turn in flatten_sample(sample())]
    formatted = format_evidence_for_answerer(turns, "2023-05-03", question_type="multi-session")
    assert "<COUNT_AND_LIST_CHECK>" in formatted.text
    assert "Deduplicate repeated mentions" in formatted.text
    assert "<RECALLED_MEMORY>" in formatted.text


def test_provence_pruner_replaces_and_drops_turn_content():
    class FakeProvence:
        def process(self, question, context, **kwargs):
            assert question == "Where is the notebook?"
            assert context == [["The notebook is in the desk. Irrelevant chatter.", "No useful sentence."]]
            assert kwargs["title"] is None
            assert kwargs["always_select_title"] is False
            return {"pruned_context": [["The notebook is in the desk.", ""]]}

    turns = [turn.to_dict() for turn in flatten_sample(sample())[2:4]]
    turns[0]["content"] = "The notebook is in the desk. Irrelevant chatter."
    turns[1]["content"] = "No useful sentence."
    pruner = ProvenceEvidencePruner(model=FakeProvence())
    result = pruner.prune_turns("Where is the notebook?", turns)

    assert [turn["stable_turn_id"] for turn in result.turns] == [turns[0]["stable_turn_id"]]
    assert result.turns[0]["content"] == "The notebook is in the desk."
    assert result.source_turn_count == 2
    assert result.kept_turn_count == 1
    assert result.output_tokens < result.input_tokens


def test_retrieval_metrics_session_hit():
    s = sample()
    turns = [turn.to_dict() for turn in flatten_sample(s)]
    matched = [
        {
            "session_id": "s2",
            "has_answer": True,
        }
    ]
    metrics = compute_retrieval_metrics(s, matched, [turns[2]], evidence_token_count=10)
    assert metrics["session_recall_at_k"] is True
    assert metrics["turn_recall_at_k"] is True
    assert metrics["expanded_turn_recall_at_k"] is True


def test_bm25_scores_keyword_match():
    index = BM25Index(["plain recipe ideas", "peanut allergy warning"])
    scores = index.scores("peanut allergy")
    assert scores[1] > scores[0]


def test_keyword_query_bm25_ignores_generic_question_terms():
    assert keyword_query_terms("What did I say today about peanut allergy?") == ["peanut", "allergy"]
    index = BM25Index(["today said ask remember", "peanut allergy warning"])
    scores = index.scores("What did I say today about peanut allergy?")
    assert scores[1] > scores[0]


def test_hybrid_ranker_ignores_zero_bm25_ties():
    raw_turns = [turn.to_dict() for turn in flatten_sample(sample())]
    dense_scores = np.asarray([0.1, 0.9, 0.2, 0.3], dtype=np.float32)
    indices, _ranked_scores, bm25_scores = rank_indices(
        query="zzzz unmatched",
        dense_scores=dense_scores,
        raw_turns=raw_turns,
        retrieval_texts=[turn["content"] for turn in raw_turns],
        candidate_indices=list(range(len(raw_turns))),
        top_k=1,
        retrieval_method="hybrid",
    )
    assert bm25_scores is not None
    assert not bm25_scores.any()
    assert indices == [1]


def test_semantic_bm25_boost_falls_back_to_dense_without_bm25_hits():
    raw_turns = [turn.to_dict() for turn in flatten_sample(sample())]
    dense_scores = np.asarray([0.1, 0.9, 0.2, 0.3], dtype=np.float32)
    diagnostics: dict[str, object] = {}
    indices, _ranked_scores, bm25_scores = rank_indices(
        query="zzzz unmatched",
        dense_scores=dense_scores,
        raw_turns=raw_turns,
        retrieval_texts=[turn["content"] for turn in raw_turns],
        candidate_indices=list(range(len(raw_turns))),
        top_k=1,
        retrieval_method="semantic_bm25_boost",
        ranking_diagnostics=diagnostics,
    )
    assert bm25_scores is not None
    assert not bm25_scores.any()
    assert indices == [1]
    assert diagnostics["lexical_rescue_count"] == 0


def test_semantic_bm25_boost_promotes_strong_keyword_hit_in_dense_pool():
    raw_turns = [
        {"session_date": "2023-05-01", "role": "user"},
        {"session_date": "2023-05-01", "role": "user"},
    ]
    indices, ranked_scores, _ = rank_indices(
        query="peanut allergy",
        dense_scores=np.asarray([0.6, 0.59], dtype=np.float32),
        raw_turns=raw_turns,
        retrieval_texts=["plain recipe ideas", "peanut allergy warning"],
        candidate_indices=[0, 1],
        top_k=2,
        retrieval_method="semantic_bm25_boost",
    )
    assert indices[0] == 1
    assert ranked_scores[1] > ranked_scores[0]


def test_semantic_bm25_boost_limits_lexical_rescue():
    raw_turns = [{"session_date": "2023-05-01", "role": "user"} for _ in range(100)]
    dense_scores = np.asarray([1.0 - idx * 0.001 for idx in range(100)], dtype=np.float32)
    retrieval_texts = ["plain text" for _ in range(100)]
    for idx in range(90, 100):
        retrieval_texts[idx] = "rareword"
    diagnostics: dict[str, object] = {}
    indices, _ranked_scores, _ = rank_indices(
        query="rareword",
        dense_scores=dense_scores,
        raw_turns=raw_turns,
        retrieval_texts=retrieval_texts,
        candidate_indices=list(range(100)),
        top_k=10,
        retrieval_method="semantic_bm25_boost",
        ranking_diagnostics=diagnostics,
    )
    rescued = [idx for idx in indices if idx >= 90]
    assert 0 < len(rescued) <= 2
    assert diagnostics["lexical_rescue_count"] == 2


def test_semantic_bm25_boost_disables_rescue_for_broad_lexical_hits():
    raw_turns = [{"session_date": "2023-05-01", "role": "user", "speaker": "A"} for _ in range(100)]
    dense_scores = np.asarray([1.0 - idx * 0.001 for idx in range(100)], dtype=np.float32)
    diagnostics: dict[str, object] = {}
    indices, _ranked_scores, _ = rank_indices(
        query="common",
        dense_scores=dense_scores,
        raw_turns=raw_turns,
        retrieval_texts=["common text" for _ in range(100)],
        candidate_indices=list(range(100)),
        top_k=10,
        retrieval_method="semantic_bm25_boost",
        ranking_diagnostics=diagnostics,
    )
    assert indices == list(range(10))
    assert diagnostics["lexical_rescue_count"] == 0
    assert diagnostics["bm25_weight"] == 0.1


def test_semantic_bm25_boost_strips_speaker_prefix_from_bm25_text():
    raw_turns = [{"session_date": "2023-05-01", "role": "user", "speaker": "Caroline"}]
    _indices, _ranked_scores, bm25_scores = rank_indices(
        query="Caroline",
        dense_scores=np.asarray([0.0], dtype=np.float32),
        raw_turns=raw_turns,
        retrieval_texts=["Caroline: plain utterance"],
        candidate_indices=[0],
        top_k=1,
        retrieval_method="semantic_bm25_boost",
    )
    assert bm25_scores is not None
    assert bm25_scores[0] == 0


def test_semantic_bm25_boost_temporal_boost_requires_current_cue():
    raw_turns = [
        {"session_date": "2023/01/01 (Sun) 09:00", "role": "user"},
        {"session_date": "2023/05/01 (Mon) 09:00", "role": "user"},
    ]
    common = {
        "dense_scores": np.array([0.0, 0.0]),
        "raw_turns": raw_turns,
        "retrieval_texts": ["old event", "new event"],
        "candidate_indices": [0, 1],
        "top_k": 2,
        "retrieval_method": "semantic_bm25_boost",
        "question_type": "temporal-reasoning",
        "question_date": "2023/06/01 (Thu) 09:00",
        "temporal_boost": 1.0,
    }
    current_indices, _current_scores, _ = rank_indices(query="what is my current status zzzz", **common)
    relation_indices, _relation_scores, _ = rank_indices(query="when did zzzz happen", **common)
    assert current_indices[0] == 1
    assert relation_indices[0] == 0


def test_bm25_lexical_text_excludes_metadata_fields():
    turn = flatten_sample(sample())[0].to_dict()
    texts = lexical_texts_for_turns([turn])
    assert texts == [turn["content"]]
    assert "[date:" not in texts[0]
    assert "[session:" not in texts[0]
    assert "[role:" not in texts[0]


def test_typed_facts_extract_money_profile_and_source_ids():
    turns = [turn.to_dict() for turn in flatten_sample(sample())]
    turns[0]["content"] = "My grandma's 75th birthday was inspiring. I spent $120 at the bike shop."
    facts = build_typed_facts([turns[0]])

    assert any(fact["fact_type"] == "profile_fact" and fact["value"] == 75 for fact in facts)
    assert any(fact["fact_type"] == "money_fact" and fact["amount_usd"] == 120 for fact in facts)
    assert source_turn_ids(facts) == {turns[0]["stable_turn_id"]}


def test_select_typed_facts_prefers_query_relevant_fact():
    turns = [turn.to_dict() for turn in flatten_sample(sample())]
    turns[0]["content"] = "I spent $120 on a Bell Zephyr bike helmet."
    turns[1]["content"] = "I loved reading \"Charlotte's Web\" as a kid."
    facts = build_typed_facts(turns[:2])

    selected = select_typed_facts("How much did I spend on bike expenses?", facts, limit=1)
    assert selected[0]["fact_type"] == "money_fact"
    assert "bike helmet" in selected[0]["source_quote"]


def test_prepend_typed_fact_pack_keeps_recalled_memory():
    turns = [turn.to_dict() for turn in flatten_sample(sample())]
    turns[0]["content"] = "I spent $120 on a Bell Zephyr bike helmet."
    facts = select_typed_facts("How much did I spend on bike expenses?", build_typed_facts([turns[0]]))

    text = prepend_typed_fact_pack("<RECALLED_MEMORY>x</RECALLED_MEMORY>", facts)
    assert text.startswith("<TYPED_FACT_MEMORY>")
    assert "Bell Zephyr" in text
    assert "<RECALLED_MEMORY>x</RECALLED_MEMORY>" in text


def test_typed_fact_routing_is_conservative():
    assert should_use_typed_facts("Which gift did I buy first?", "temporal-reasoning")
    assert should_use_typed_facts("How much did I spend on bike expenses?", "multi-session")
    assert should_use_typed_facts("Where has Melanie camped?", "multi-hop")
    assert not should_use_typed_facts("What did Caroline do yesterday?", "multi-hop")


def test_temporal_boost_prefers_recent_metadata_for_temporal_question():
    raw_turns = [
        {"session_date": "2023/01/01 (Sun) 09:00"},
        {"session_date": "2023/05/01 (Mon) 09:00"},
    ]
    indices, ranked_scores, _ = rank_indices(
        query="what happened recently?",
        dense_scores=np.array([0.0, 0.0]),
        raw_turns=raw_turns,
        retrieval_texts=["old event", "new event"],
        candidate_indices=[0, 1],
        top_k=2,
        question_type="temporal-reasoning",
        question_date="2023/06/01 (Thu) 09:00",
        temporal_boost=1.0,
    )
    assert indices[0] == 1
    assert ranked_scores[1] > ranked_scores[0]


def test_timestamp_filter_excludes_future_sessions():
    raw_turns = [
        {"session_date": "2023-05-03 18:00", "session_id": "same-day"},
        {"session_date": "2023-05-04 09:00", "session_id": "future"},
        {"session_date": "2023-05-02 09:00", "session_id": "past"},
    ]
    indices = _filter_indices(raw_turns, None, "2023-05-03")
    assert indices == [0, 2]


def test_token_accounting_separates_build_query_judge():
    stats = new_token_summary()
    add_build_tokens(stats, "q1", 10, 8)
    add_query_tokens(stats, "q1", 20, 5)
    add_judge_tokens(stats, "q1", 7, 2)
    assert stats["build_tokens"]["embedding_input_tokens"] == 10
    assert stats["query_tokens"]["total_tokens"] == 25
    assert stats["judge_tokens"]["total_tokens"] == 9
    assert stats["method_cost_tokens"]["judge_total_tokens_excluded"] == 9


def test_predictions_jsonl_format(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [{"question_id": "q1", "hypothesis": "answer"}])
    line = path.read_text(encoding="utf-8").strip()
    assert json.loads(line) == {"question_id": "q1", "hypothesis": "answer"}


def test_error_analysis_filters_to_run_samples(tmp_path: Path):
    s1 = sample()
    s2 = LongMemEvalSample.from_dict({**s1.to_dict(), "question_id": "q2"})
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "error_analysis.jsonl"
    write_predictions(predictions_path, [{"question_id": "q1", "hypothesis": "answer"}])
    write_error_analysis([s1, s2], [{"question_id": "q1", "metrics": {}, "matched_turns": []}], predictions_path, new_token_summary(), output_path)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["question_id"] for row in rows] == ["q1"]


def test_likely_failure_type_reads_judge_dict_label():
    metrics = {"session_recall_at_k": True, "turn_recall_at_k": True, "expanded_turn_recall_at_k": True}
    assert likely_failure_type(sample(), metrics, {"model": "judge", "label": False}) == "possible_temporal_failure"


def test_merge_config_preserves_common_run_settings(tmp_path: Path):
    part0 = tmp_path / "part0"
    part1 = tmp_path / "part1"
    part0.mkdir()
    part1.mkdir()
    write_json(part0 / "config.json", {"run_id": "p0", "top_k": 10, "message_range": 2, "selected_question_ids": ["a"], "num_selected_samples": 1})
    write_json(part1 / "config.json", {"run_id": "p1", "top_k": 10, "message_range": 2, "selected_question_ids": ["b"], "num_selected_samples": 1})
    config = _merge_config([part0, part1], "merged")
    assert config["run_id"] == "merged"
    assert config["top_k"] == 10
    assert config["message_range"] == 2
    assert config["source_run_ids"] == ["p0", "p1"]
    assert config["selected_question_ids"] == ["a", "b"]


def test_locomo_judge_prompt_uses_requested_template():
    assert "Your task is to label an answer to a question" in LOCOMO_JUDGE_PROMPT
    assert 'key as "label"' in LOCOMO_JUDGE_PROMPT
    assert "Do NOT include both CORRECT and WRONG" in LOCOMO_JUDGE_PROMPT
    assert "DATE TOLERANCE" not in LOCOMO_JUDGE_PROMPT


def test_locomo_retrieval_result_can_be_sliced_by_topk():
    turns = [turn.to_dict() for turn in flatten_sample(sample())]
    for idx, turn in enumerate(turns):
        turn["score"] = 1.0 - idx * 0.1
        turn["rank"] = idx + 1
    row = {
        "question_id": "q1",
        "question_type": "temporal-reasoning",
        "question_date": "2023-05-03",
        "answer_session_ids": ["s2"],
        "matched_turns": turns,
        "evidence_windows": [{"turns": [turn]} for turn in turns],
    }
    sliced = _slice_locomo_retrieval_result(row, 3, None)
    assert sliced["top_k"] == 3
    assert len(sliced["matched_turns"]) == 3
    assert sliced["metrics"]["session_recall_at_k"] is True
    assert sliced["metrics"]["expanded_turn_recall_at_k"] is True


def test_longmemeval_shard_selection_is_deterministic():
    samples = [
        LongMemEvalSample.from_dict({**sample().to_dict(), "question_id": f"q{i}"})
        for i in range(5)
    ]
    assert [item.question_id for item in _apply_shard(samples, 2, 0)] == ["q0", "q2", "q4"]
    assert [item.question_id for item in _apply_shard(samples, 2, 1)] == ["q1", "q3"]


def test_longmemeval_retrieval_result_can_be_sliced_by_topk():
    turns = [turn.to_dict() for turn in flatten_sample(sample())]
    matched_turns = []
    for idx, turn in enumerate(turns):
        matched_turns.append(
            {
                "rank": idx + 1,
                "stable_turn_id": turn["stable_turn_id"],
                "score": 1.0 - idx * 0.1,
                "session_id": turn["session_id"],
                "session_date": turn["session_date"],
                "session_idx": turn["session_idx"],
                "turn_idx": turn["turn_idx"],
                "role": turn["role"],
                "content": turn["content"],
                "has_answer": turn["has_answer"],
            }
        )
    row = {
        "question_id": "q1",
        "question_type": "temporal-reasoning",
        "question_date": "2023-05-03",
        "answer_session_ids": ["s2"],
        "matched_turns": matched_turns,
        "evidence_windows": [
            {"matched_stable_turn_id": turn["stable_turn_id"], "turns": [turn]}
            for turn in turns
        ],
    }
    sliced = _slice_longmemeval_retrieval_result(row, 3, None)
    assert sliced["top_k"] == 3
    assert len(sliced["matched_turns"]) == 3
    assert sliced["metrics"]["session_recall_at_k"] is True
    assert sliced["metrics"]["expanded_turn_recall_at_k"] is True


def test_llm_cache_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    body = {"choices": [{"message": {"content": "hello"}}], "usage": {"total_tokens": 3}}
    assert cached_response("answer", "m", payload) is None
    write_cached_response("answer", "m", payload, body)
    assert cached_response("answer", "m", payload) == body
