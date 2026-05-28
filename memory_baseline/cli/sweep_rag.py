from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from memory_baseline.core.env import load_project_env


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag_baseline_longmemeval.yaml")
    parser.add_argument("--top-k-list", default=None)
    parser.add_argument("--message-range-list", default=None)
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--mode", choices=["retrieval-only", "full"], default="retrieval-only")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    top_k_list = parse_int_list(args.top_k_list) if args.top_k_list else config.get("top_k_list", [20])
    message_range_list = (
        parse_int_list(args.message_range_list) if args.message_range_list else config.get("message_range_list", [2])
    )
    question_limit = args.question_limit if args.question_limit is not None else config.get("question_limit", 20)

    for top_k in top_k_list:
        for message_range in message_range_list:
            run_id = f"rag_turn_top{top_k}_mr{message_range}_n{question_limit}"
            common = [
                sys.executable,
                "-m",
                "memory_baseline.run_longmemeval",
                "--data",
                str(config.get("data", "data/longmemeval_s_cleaned.json")),
                "--run-id",
                run_id,
                "--top-k",
                str(top_k),
                "--message-range",
                str(message_range),
                "--chunk-mode",
                str(config.get("chunk_mode", "turn")),
                "--question-limit",
                str(question_limit),
                "--output-dir",
                str(config.get("output_dir", "runs")),
                "--skip-existing",
            ]
            if config.get("embedding_model"):
                common.extend(["--embedding-model", str(config["embedding_model"])])
            if config.get("embedding_backend"):
                common.extend(["--embedding-backend", str(config["embedding_backend"])])
            if config.get("answer_model"):
                common.extend(["--answer-model", str(config["answer_model"])])
            if config.get("answer_base_url"):
                common.extend(["--answer-base-url", str(config["answer_base_url"])])
            if args.mode == "full":
                subprocess.run(common + ["--mode", "full"], check=True)
            else:
                for stage in ["build", "retrieve", "eval-retrieval"]:
                    subprocess.run(common + ["--mode", stage], check=True)


if __name__ == "__main__":
    main()
