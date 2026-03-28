---
pretty_name: OmniCX Logistics CX Dataset
language:
  - en
license: cc-by-4.0
task_categories:
  - text-generation
  - text-classification
task_ids:
  - text2text-generation
size_categories:
  - n<1K
---

# OmniCX Logistics CX Dataset (Research Preview)

## Table of Contents

- [Dataset Summary](#dataset-summary)
- [Supported Tasks](#supported-tasks)
- [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Schema Highlights](#schema-highlights)
- [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Source Data](#source-data)
  - [Knowledge-Source Derivation (Important)](#knowledge-source-derivation-important)
  - [Annotation Process](#annotation-process)
  - [Quality Control](#quality-control)
- [Limitations](#limitations)
- [Bias, Risks, and Safety](#bias-risks-and-safety)
- [Recommended Uses](#recommended-uses)
- [Out-of-Scope Uses](#out-of-scope-uses)
- [Licensing](#licensing)
- [Citation](#citation)

## Dataset Summary

This dataset is designed for structured extraction of logistics and customer-experience (CX) signals from multi-turn support conversations.

Each record uses ChatML-style messages with:
- a fixed `system` instruction
- a `user` transcript
- an `assistant` JSON payload matching `LogisticsCXMetrics`

This release is a **research preview** and should not be treated as a production-certified benchmark.

Project repository: [OmniCX-Extractor](https://github.com/mangesh-ux/OmniCX-Extractor)

### Taxonomy Summary

`LogisticsCXMetrics` contains three top-level groups:
- **`behavioral_analytics`**: intent, effort (`1-5` CES-like rubric), sentiment trajectory, rework frequency, and friction evidence quotes.
- **`operational_analytics`**: exception diagnosis, controlled exception/root-cause categories, deterministic boolean flags, and resolution tracking.
- **`diagnostic_reasoning`**: auditable reasoning fields (`intent_reasoning`, `exception_reasoning`, `effort_reasoning`) plus routing recommendation.

Core controlled vocabularies include customer intent families, rework bands (`0`, `1`, `2+`), sentiment trajectory (`Improved`, `Worsened`, `Unchanged`), and root-cause families.

## Supported Tasks

- Structured information extraction from support transcripts
- Multi-label analytics extraction
- Schema-constrained generation

## Languages

- English (`en`)

## Dataset Structure

### Data Instances

Each row is one JSON object:

```json
{
  "messages": [
    {"role": "system", "content": "You are a SOTA Logistics AI. Extract the exact logistics and CX metrics from the following transcript."},
    {"role": "user", "content": "Agent: ... Customer: ..."},
    {"role": "assistant", "content": "{\"behavioral_analytics\": {...}, \"operational_analytics\": {...}, \"diagnostic_reasoning\": {...}}"}
  ]
}
```

### Schema Highlights

The assistant JSON contains three required sections:
- `behavioral_analytics`
- `operational_analytics`
- `diagnostic_reasoning`

Field definitions and enums are implemented in `src/schema.py`.
Detailed taxonomy/rubric reference:
- [`docs/taxonomy.md`](https://github.com/mangesh-ux/OmniCX-Extractor/blob/main/docs/taxonomy.md)

## Data Splits

Current repository artifacts include:
- training JSONL in `data/processed/`
- evaluation JSONL in `data/eval/`

For Hugging Face release, publish explicit split files:
- `train.jsonl`
- `validation.jsonl` (optional but recommended)
- `test.jsonl`

## Dataset Creation

### Source Data

This project primarily uses synthetic logistics support conversations and synthetic labels generated through controlled prompting and schema validation.

Synthetic generation models used in this repository:
- Transcript generation: `gpt-4o-mini` (`src/data_factory.py`)
- Schema-constrained label extraction: `gpt-4o-mini` (`src/extractor.py`)

### Knowledge-Source Derivation (Important)

The output structure and taxonomy are derived from curated reference material in `docs/knowledge/`, including:
- `Transcript-Only CX Difficulty Score_ Standards, Methods, and a Rigorous MVP Design.pdf`  
  Deep-research document (ChatGPT-generated) focused on transcript-only CX friction and effort signals, including rework, escalation cues, sentiment volatility, unresolved follow-up markers, and rubric design for difficulty/effort estimation.
- `Logistics CX Data Schema Development.docx`  
  NotebookLM-assisted research and design artifact focused on logistics intent taxonomy and schema structuring, used to refine intent families, enum boundaries, and extraction-ready field definitions.

Field definitions, enum choices, and diagnostic categories in assistant JSON are grounded in these source documents and enforced through `LogisticsCXMetrics` validation (`src/schema.py`).

### Annotation Process

Labels are represented as structured JSON targeting the `LogisticsCXMetrics` schema and include behavioral, operational, and reasoning components.

### Quality Control

- format validation for JSONL integrity
- required-key checks for schema completeness
- parseability checks for assistant JSON content
- iterative cleanup scripts for malformed examples

## Limitations

- Small dataset size in current iteration
- Distribution mismatch risk versus real support logs
- Strict exact-match scoring may understate semantically-correct outputs
- Not calibrated for legal/compliance decisions

## Bias, Risks, and Safety

- Synthetic generation may encode stylistic bias from prompting models
- Root-cause and effort labels can reflect rubric bias
- Outputs should be human-reviewed for operational actions
- Not intended for automated denial/escalation adjudication

## Recommended Uses

- Research on schema-constrained extraction
- Prototyping CX analytics pipelines
- Error analysis and model behavior studies

## Out-of-Scope Uses

- Fully autonomous customer adjudication
- Legal/regulatory decisions without human oversight
- Claims/payment decision automation

## Licensing

This card is written assuming `CC-BY-4.0` for dataset artifacts. Confirm and publish your final legal choice in both:
- dataset repo license metadata
- repository `LICENSE` file

## Citation

```bibtex
@dataset{omnicx_logistics_cx_preview,
  title = {OmniCX Logistics CX Dataset (Research Preview)},
  author = {Mangesh Gupta},
  year = {2026},
  publisher = {Hugging Face},
  note = {Synthetic logistics CX extraction dataset}
}
```
