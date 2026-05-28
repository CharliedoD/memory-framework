from __future__ import annotations

import os

from memory_baseline.core.env import load_project_env


def test_load_project_env_reads_dotenv_without_override(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "ANSWERER_API_KEY=from_file",
                "ANSWERER_MODEL='model-name'",
                "EMPTY_VALUE=",
                "export JUDGE_MODEL=judge-name # comment",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANSWERER_API_KEY", "from_env")
    monkeypatch.delenv("ANSWERER_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)

    load_project_env(env_path)

    assert os.environ["ANSWERER_API_KEY"] == "from_env"
    assert os.environ["ANSWERER_MODEL"] == "model-name"
    assert os.environ["JUDGE_MODEL"] == "judge-name"
    assert os.environ["EMPTY_VALUE"] == ""
