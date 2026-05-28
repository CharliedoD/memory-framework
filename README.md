# memory-framework

Clean Raw-RAG memory baselines for LongMemEval-style experiments.

This repository keeps only the reproducible baseline code, configs, and tests. Local experiment outputs, caches, logs, docs, and benchmark datasets are intentionally excluded from git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[hf]"
```

For a lightweight environment without local HuggingFace embeddings:

```bash
pip install -e .
```

## Environment

```bash
cp .env.example .env
```

Fill in API keys in `.env` if you run answer or judge stages. Retrieval smoke tests can use the built-in `local-hash` embedder without API keys.

## Data

Benchmark data is not committed. Put datasets under `data/`, for example:

```text
data/longmemeval/longmemeval_s_cleaned.json
data/locomo/locomo_dev_conv0.json
```

`tests/fixtures/longmemeval_s_tiny.json` is included only for local tests and smoke checks.

## Quick Checks

```bash
pytest
python -m memory_baseline.cli.run_longmemeval \
  --data tests/fixtures/longmemeval_s_tiny.json \
  --run-id smoke_localhash \
  --embedding-model local-hash \
  --mode index \
  --top-k 5
```

Full answer and judge runs require model endpoints configured in `.env`.

## Provence Pruning

After build/retrieve has produced `retrieval_results.jsonl` for the same run id, run the answer stage without the evidence compiler and prune retrieved turn text with Provence before answer prompting:

```bash
python -m memory_baseline.cli.run_longmemeval \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --run-id rag_provence_no_compiler \
  --mode answer \
  --disable-evidence-compiler \
  --provence-pruning
```

Install the HuggingFace extras and the NLTK sentence model first:

```bash
pip install -e ".[hf]"
python -c "import nltk; nltk.download('punkt_tab')"
```
