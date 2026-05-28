from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from memory_baseline.core.env import load_project_env


def official_judge_command(
    longmemeval_dir: str | Path,
    judge_model: str,
    predictions_file: str | Path,
    data_file: str | Path,
) -> tuple[Path, list[str]]:
    eval_dir = Path(longmemeval_dir) / "src" / "evaluation"
    return eval_dir, ["python3", "evaluate_qa.py", judge_model, str(Path(predictions_file).resolve()), str(Path(data_file).resolve())]


def run_official_judge(longmemeval_dir: str | Path, judge_model: str, predictions_file: str | Path, data_file: str | Path) -> None:
    load_project_env()
    eval_dir, command = official_judge_command(longmemeval_dir, judge_model, predictions_file, data_file)
    if not (eval_dir / "evaluate_qa.py").exists():
        raise FileNotFoundError(f"LongMemEval evaluate_qa.py not found under {eval_dir}")
    subprocess.run(command, cwd=eval_dir, check=True)


def main(argv: list[str] | None = None) -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--longmemeval-dir", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--data-file", required=True)
    args = parser.parse_args(argv)
    run_official_judge(args.longmemeval_dir, args.judge_model, args.predictions_file, args.data_file)


if __name__ == "__main__":
    main()
