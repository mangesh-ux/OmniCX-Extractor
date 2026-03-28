---
language:
  - en
license: apache-2.0
pipeline_tag: text-generation
tags:
  - logistics
  - customer-support
  - information-extraction
  - qlora
  - unsloth
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: transformers
---

# OmniCX Qwen2.5-3B LoRA (Research Preview)

## Table of Contents

- [Model Description](#model-description)
- [Model Details](#model-details)
- [Training Data](#training-data)
- [Training Procedure](#training-procedure)
- [Evaluation](#evaluation)
- [Intended Uses](#intended-uses)
- [Out-of-Scope Uses](#out-of-scope-uses)
- [Limitations](#limitations)
- [Bias, Risks, and Safety](#bias-risks-and-safety)
- [How to Use](#how-to-use)
  - [Input and Output Contract](#input-and-output-contract)
- [Versioning](#versioning)
- [Citation](#citation)

## Model Description

This model is a QLoRA fine-tune of `Qwen/Qwen2.5-3B-Instruct` for extracting structured logistics and customer-experience analytics from support transcripts.

The target output is a strict JSON object compatible with `LogisticsCXMetrics` (`behavioral_analytics`, `operational_analytics`, `diagnostic_reasoning`).

The output schema and taxonomy are derived from curated reference files:
- [`Transcript-Only CX Difficulty Score_ Standards, Methods, and a Rigorous MVP Design.pdf`](https://github.com/mangesh-ux/OmniCX-Extractor/blob/main/docs/knowledge/Transcript-Only%20CX%20Difficulty%20Score_%20Standards%2C%20Methods%2C%20and%20a%20Rigorous%20MVP%20Design.pdf)  
  Deep-research document (ChatGPT-generated) on transcript-only CX friction signals and effort scoring methodology.
- [`Logistics CX Data Schema Development.docx`](https://github.com/mangesh-ux/OmniCX-Extractor/blob/main/docs/knowledge/Logistics%20CX%20Data%20Schema%20Development.docx)  
  NotebookLM-assisted intent and schema research used to shape intent taxonomy and extraction field design.

These definitions are operationalized in `src/schema.py` and reflected in training labels.
Canonical taxonomy and rubric reference:
- [`docs/taxonomy.md`](https://github.com/mangesh-ux/OmniCX-Extractor/blob/main/docs/taxonomy.md)

This release is a **research preview**, not a production-certified model.

Project repository: [OmniCX-Extractor](https://github.com/mangesh-ux/OmniCX-Extractor)

## Model Details

- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Fine-tuning method:** QLoRA (4-bit) via Unsloth
- **Adapter format:** LoRA adapter
- **Primary use case:** structured extraction for logistics CX research workflows

## Training Data

- Main training artifact: `data/processed/golden_training_dataset.jsonl`
- Sample size (iteration shown): 486 examples
- Data format: ChatML-style messages with assistant JSON labels
- Label space source: `docs/knowledge/` references (field/taxonomy source), mapped to `LogisticsCXMetrics`

## Training Procedure

- Max sequence length: 2048
- Total steps: 150
- Effective batch size: 8
- Learning rate: 2e-4 (linear schedule)
- Optimizer: `adamw_8bit`
- Environment: single 8GB VRAM GPU setup (see training logs)

Detailed run record:
- [`docs/training_logs/iteration_001.md`](https://github.com/mangesh-ux/OmniCX-Extractor/blob/main/docs/training_logs/iteration_001.md)

## Evaluation

Current evaluation (research preview):

- Eval examples: 32
- Runtime errors: 0
- Strict exact-match accuracy: 0.0% (0/32)
- Mean latency: 29.84s/sample
- Min / max latency: 16.89s / 45.72s
- Total latency: 954.94s

Selected per-field accuracy:
- `customer_intent`: 56.2%
- `sentiment_trajectory`: 65.6%
- `address_change_requested`: 100.0%
- `escalation_requested`: 100.0%

Detailed report:
- [`eval_report_iteration_001.md`](./eval_report_iteration_001.md)
- [`eval_outputs_iteration_001.jsonl`](./eval_outputs_iteration_001.jsonl)

## Intended Uses

- Research and prototyping for logistics transcript understanding
- Structured extraction experiments under human review
- Error analysis and taxonomy tuning

## Out-of-Scope Uses

- Autonomous production decisioning without human review
- Legal, financial, or regulatory adjudication
- High-risk customer-impacting automation

## Limitations

- Small current eval set and strict metric sensitivity
- Potential mismatch to real-world transcript distribution
- Schema-conformant generation is not guaranteed in all cases

## Bias, Risks, and Safety

- Synthetic or rubric-driven labels can encode design bias
- Output confidence is not calibrated for risk-critical decisions
- Use human oversight for escalations and customer-impacting actions

## How to Use

### Load adapter and run extraction (project-local)

```python
from src.inference import load_model, extract_with_finetuned

model, tokenizer = load_model(model_path="models/qwen-logistics-lora")
result = extract_with_finetuned(
    transcript="Agent: ... Customer: ...",
    model=model,
    tokenizer=tokenizer,
    return_dict=True,
)
print(result)
```

### Download from Hugging Face and run locally

```python
from huggingface_hub import snapshot_download
from src.inference import load_model, extract_with_finetuned

local_model_dir = snapshot_download("mangesh-ux/omnicx-logistics-cx-extractor-qwen25-3b-lora")
model, tokenizer = load_model(model_path=local_model_dir)
result = extract_with_finetuned(
    transcript="Agent: ... Customer: ...",
    model=model,
    tokenizer=tokenizer,
    return_dict=True,
)
print(result)
```

### Input and Output Contract

**Input (single transcript):**

```json
{
  "transcript": "Agent: ... Customer: ..."
}
```

**Output (schema-aligned JSON):**

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

The full field contract and enums are defined in `src/schema.py`.

## Versioning

Recommended release naming:
- `v0.1.0` - initial research preview
- `v0.1.1+` - format, eval, and quality refinements

## Citation

```bibtex
@misc{omnicx_qwen25_lora_preview,
  title = {OmniCX Qwen2.5-3B LoRA (Research Preview)},
  author = {Mangesh Gupta},
  year = {2026},
  publisher = {Hugging Face},
  note = {QLoRA fine-tune for logistics CX structured extraction}
}
```
