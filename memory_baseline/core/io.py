from __future__ import annotations

from pathlib import Path

from .utils import read_jsonl, write_jsonl


def write_predictions(path: str | Path, predictions: list[dict[str, str]]) -> None:
    records = [
        {"question_id": str(prediction["question_id"]), "hypothesis": str(prediction["hypothesis"])}
        for prediction in predictions
    ]
    write_jsonl(path, records)


def read_predictions(path: str | Path) -> dict[str, str]:
    return {record["question_id"]: record.get("hypothesis", "") for record in read_jsonl(path)}
