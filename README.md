# OmniCX Extractor

Structured logistics and customer-experience intelligence from raw support conversations.

OmniCX Extractor fine-tunes `Qwen/Qwen2.5-3B-Instruct` (QLoRA + Unsloth) to convert multi-turn chat/call/email-style transcripts into a strict JSON schema (`LogisticsCXMetrics`) for downstream analytics.

## Why This Project Exists

Logistics support interactions contain both:
- **Behavioral CX signals** (effort, friction, sentiment trajectory, escalation)
- **Operational signals** (delivery exceptions, root cause category, resolution state)

Most pipelines optimize for one side. OmniCX Extractor is built to capture both in a single structured output.

## What Is Shipped

- Schema-first extraction in `src/schema.py`
- Fine-tuning pipeline in `src/train.py`
- Inference utilities + local API (`src/inference.py`, `src/serve_inference.py`)
- Evaluation pipeline with per-field metrics, strict match, and latency (`scripts/run_evaluation.py`)
- Dataset/model release utilities for Hugging Face (`scripts/prepare_hf_release.py`, `scripts/publish_hf_artifacts.py`)

## Current Status (Iteration 001)

- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Finetuning:** QLoRA (4-bit) via Unsloth
- **Training samples:** 486
- **Eval samples:** 32 (cleaned, parse-valid)
- **Strict exact-match accuracy:** 0.0% (0/32)
- **Mean latency:** 29.84s per example

Detailed artifacts:
- `docs/training_logs/iteration_001.md`
- `docs/training_logs/eval_report_iteration_001.md`
- `docs/training_logs/eval_outputs_iteration_001.jsonl`

## Published Artifacts

- Hugging Face dataset (research preview): [mangesh-ux/logistics-cx-transcript-analysis-chatml](https://huggingface.co/datasets/mangesh-ux/logistics-cx-transcript-analysis-chatml)
- Hugging Face model (research preview): [mangesh-ux/omnicx-logistics-cx-extractor-qwen25-3b-lora](https://huggingface.co/mangesh-ux/omnicx-logistics-cx-extractor-qwen25-3b-lora)

## Knowledge-Grounded Schema

Output fields and taxonomies are grounded in the research/source docs in `docs/knowledge/`, including:
- `Transcript-Only CX Difficulty Score_ Standards, Methods, and a Rigorous MVP Design.pdf`
- `Logistics CX Data Schema Development.docx`

These are mapped into `LogisticsCXMetrics` and enforced during data prep, training, and evaluation.

Canonical taxonomy and rubric spec:
- `docs/taxonomy.md`

### Synthetic Data Generation Models

Synthetic data creation pipeline (`src/data_factory.py`) currently uses:
- **Transcript generation model:** `gpt-4o-mini`
- **Label extraction model (schema-constrained):** `gpt-4o-mini` via `src/extractor.py`

These model choices apply to the synthetic training data generation flow in this repository and should be updated in docs/cards if changed in future iterations.

### Taxonomy Overview

`LogisticsCXMetrics` contains three top-level groups:
- **`behavioral_analytics`**: customer intent, effort (`1-5` CES-like rubric), sentiment trajectory, rework frequency, and direct friction quotes.
- **`operational_analytics`**: exception diagnosis, `delivery_exception_type`, `root_cause_category`, deterministic operational flags, and explicit resolution state.
- **`diagnostic_reasoning`**: auditable reasoning text (`intent_reasoning`, `exception_reasoning`, `effort_reasoning`) plus recommended routing queue.

Key controlled vocabularies include:
- **Intent taxonomy**: `WISMO_Standard`, `Address_Modification`, `Proof_of_Delivery_Dispute`, `Damage_Claim_Initiation`, `Lost_in_Transit_Investigation`, etc.
- **Rework taxonomy**: `0`, `1`, `2+` repetition bands.
- **Sentiment trajectory taxonomy**: `Improved`, `Worsened`, `Unchanged`.
- **Root-cause families**: Address/Recipient, Environmental/Force Majeure, Operational/Mechanical/Technological, Documentation/Labeling, Hazmat, Unknown.

Default ambiguity policy:
- Use `Unknown / Not Explicitly Stated` when exception evidence is insufficient.
- Use `Unknown / Not Applicable` when root cause cannot be grounded in transcript evidence.
- Set `agent_explicitly_confirmed_resolution = true` only when explicitly confirmed in the interaction.

## Repository Layout

```text
OmniCX-Extractor/
  src/
    schema.py
    train.py
    inference.py
    serve_inference.py
    data_factory.py
  scripts/
    run_evaluation.py
    fix_eval_jsonl.py
    filter_valid_eval_lines.py
    prepare_hf_release.py
    publish_hf_artifacts.py
    upload_dataset_hf.py
  docs/
    eval/
    hf/
    knowledge/
    training_logs/
  data/
    processed/
    eval/
  models/
    qwen-logistics-lora/
  hf_release/
    dataset/
    model/
```

## Data Format (ChatML JSONL)

Each line is one sample:

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

For GPU training, use the environment where CUDA-enabled PyTorch and Unsloth are installed (WSL + conda recommended for this project).

### 2) Train

```bash
python src/train.py
```

### 3) Evaluate

```bash
python scripts/run_evaluation.py \
  --eval-file data/eval/eval_dataset.jsonl \
  --report docs/training_logs/eval_report_iteration_001.md \
  --results-json docs/training_logs/eval_results_iteration_001.json \
  --outputs docs/training_logs/eval_outputs_iteration_001.jsonl
```

### 4) Serve locally

```bash
python src/serve_inference.py
```

Test request:

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d "{\"transcript\":\"Agent: Hi. Customer: Where is my order?\"}"
```

## End-User Input and Output

### Input contract

Minimum request payload:

```json
{
  "transcript": "Agent: ... Customer: ..."
}
```

Optional payload with prompt override:

```json
{
  "transcript": "Agent: ... Customer: ...",
  "system_prompt": "You are a logistics analyst. Extract metrics."
}
```

### Output contract

Service returns structured JSON with three top-level groups:

```json
{
  "behavioral_analytics": {
    "customer_intent": "WISMO_Standard",
    "customer_effort_score": 2
  },
  "operational_analytics": {
    "delivery_exception_type": "Unknown / Not Explicitly Stated",
    "root_cause_category": "Unknown / Not Applicable",
    "agent_explicitly_confirmed_resolution": true
  },
  "diagnostic_reasoning": {
    "recommended_routing_queue": "Tier 1 Support"
  }
}
```

For complete field definitions and enums, see `src/schema.py`.

## Release Workflow (Hugging Face)

```bash
python scripts/prepare_hf_release.py
python scripts/upload_dataset_hf.py --repo mangesh-ux/logistics-cx-transcript-analysis-chatml
# or publish both dataset + model:
# python scripts/publish_hf_artifacts.py --dataset-repo ... --model-repo ...
```

### HF API Smoke Test

Run this to verify model + dataset Hub APIs are healthy before/after publish:

```bash
python scripts/smoke_test_hf_apis.py
```

What it validates:
- model metadata + required files
- dataset metadata + required files
- small-file downloadability
- dataset splits endpoint (`train` / `test`) from datasets-server

## Roadmap (Near-Term)

- Improve strict schema-level accuracy with more diverse labeled samples
- Expand data sources/personas/channels and rebalance difficult edge-cases
- Publish model artifact and versioned changelog (`v0.1.x` -> `v0.2.0`)
- Add deeper benchmark reporting (segment-level and robustness slices)