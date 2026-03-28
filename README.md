# OmniCX Extractor

Structured logistics + customer-experience intelligence from raw support conversations.

OmniCX Extractor fine-tunes `Qwen/Qwen2.5-3B-Instruct` with QLoRA (Unsloth) to convert multi-turn support transcripts into a strict, analytics-ready JSON schema (`LogisticsCXMetrics`).

## Why This Project Exists

Logistics support transcripts contain both:
- **Behavioral CX signals** (friction, effort, escalation, sentiment shift)
- **Operational failure signals** (delivery exceptions, root-cause category, resolution state)

Most pipelines capture one side and miss the other. OmniCX Extractor is built to capture both in a single pass.

## What You Get

- **Schema-first extraction** with deterministic structure in `src/schema.py`
- **Fine-tuned local model** in `models/qwen-logistics-lora/`
- **Evaluation pipeline** with per-field accuracy, strict accuracy, and latency metrics
- **Per-example outputs** (prediction + gold + latency) for auditing
- **Local API server** for integration testing (`/extract`)

## Current Snapshot (Iteration 001)

- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Fine-tuning:** QLoRA (4-bit) with Unsloth
- **Training set:** `486` examples
- **Eval set:** `32` examples (cleaned and parse-valid)
- **Strict accuracy:** `0.0%` (0/32)
- **Latency (mean):** `22.35s` per example

Detailed logs:
- `docs/training_logs/iteration_001.md`
- `docs/training_logs/eval_report_iteration_001.md`
- `docs/training_logs/eval_outputs_iteration_001.jsonl`

## Repository Layout

```text
OmniCX-Extractor/
  src/
    schema.py                # Core extraction schema (LogisticsCXMetrics)
    train.py                 # QLoRA fine-tuning script
    inference.py             # Local inference utilities
    serve_inference.py       # Flask server for /extract
    data_factory.py          # Synthetic data generation pipeline
  data/
    raw/
    processed/               # Golden training JSONL files
    eval/                    # Eval dataset JSONL
  scripts/
    run_evaluation.py        # Accuracy + latency + outputs reporting
    fix_eval_jsonl.py
    filter_valid_eval_lines.py
    plot_iteration_001_loss.py
  docs/
    eval/
    training_logs/
  models/
    qwen-logistics-lora/     # Trained adapter artifacts
```

## Data Format (ChatML JSONL)

Each line is one training/eval example:

```json
{
  "messages": [
    {"role": "system", "content": "You are a SOTA Logistics AI. Extract the exact logistics and CX metrics from the following transcript."},
    {"role": "user", "content": "Agent: ... Customer: ..."},
    {"role": "assistant", "content": "{\"behavioral_analytics\": {...}, \"operational_analytics\": {...}, \"diagnostic_reasoning\": {...}}"}
  ]
}
```

## Quickstart

### 1) Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to train/infer locally, also install your GPU stack and training deps (`torch`, `transformers`, `trl`, `datasets`, `peft`, `unsloth`, etc.) that match your CUDA setup.

### 2) Train (optional)

```bash
python src/train.py
```

Outputs are saved to:
- `outputs/`
- `models/qwen-logistics-lora/`

### 3) Run evaluation

```bash
python scripts/run_evaluation.py \
  --eval-file data/eval/eval_dataset.jsonl \
  --report docs/training_logs/eval_report_iteration_001.md \
  --results-json docs/training_logs/eval_results_iteration_001.json \
  --outputs docs/training_logs/eval_outputs_iteration_001.jsonl
```

### 4) Serve local API

```bash
python src/serve_inference.py
```

Test:

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d "{\"transcript\":\"Agent: Hi. Customer: Where is my order?\"}"
```