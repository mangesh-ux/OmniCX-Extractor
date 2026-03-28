# Hugging Face Publish Runbook (Elite Research Standard)

This checklist is optimized for a **research-preview launch** that still looks professional and reproducible.

## Phase 0 - Governance Gate (must pass)

- [ ] Add a top-level `LICENSE` file
- [ ] Confirm no secrets in tracked files or history (`.env`, `openai_api_key.txt`, tokens)
- [ ] Confirm data is safe to publish (no PII/proprietary raw customer records)
- [ ] Decide release posture: `research preview` (recommended now)

## Phase 1 - Prepare artifacts

### Dataset artifacts
- [ ] Export finalized split files (recommended):
  - `train.jsonl`
  - `validation.jsonl` (optional)
  - `test.jsonl`
- [ ] Add dataset card as `README.md` in dataset release folder (start from `docs/hf/DATASET_CARD.md`)

### Model artifacts
- [ ] Include LoRA adapter files (or merged model if intentionally releasing merged weights)
- [ ] Add model card as `README.md` in model release folder (start from `docs/hf/MODEL_CARD.md`)
- [ ] Include eval report and per-example outputs as supporting artifacts

## Phase 2 - Create Hugging Face repos

> Requires: `pip install huggingface_hub` and `huggingface-cli login`

```bash
huggingface-cli whoami
```

Create repos (via UI or CLI):
- dataset repo: `your-username/omnicx-logistics-cx-dataset`
- model repo: `your-username/omnicx-qwen25-3b-lora`

## Phase 3 - Push dataset

Recommended local staging folder structure:

```text
hf_release/
  dataset/
    README.md         # from docs/hf/DATASET_CARD.md
    train.jsonl
    validation.jsonl  # optional
    test.jsonl
    LICENSE
```

Upload with Python API:

```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "your-username/omnicx-logistics-cx-dataset"
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path="hf_release/dataset",
    repo_id=repo_id,
    repo_type="dataset",
)
print("Dataset uploaded.")
```

## Phase 4 - Push model

Recommended local staging folder structure:

```text
hf_release/
  model/
    README.md         # from docs/hf/MODEL_CARD.md
    adapter_config.json
    adapter_model.safetensors
    tokenizer.json
    tokenizer_config.json
    chat_template.jinja
    LICENSE
    eval_report_iteration_001.md
```

Upload with Python API:

```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "your-username/omnicx-qwen25-3b-lora"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="hf_release/model",
    repo_id=repo_id,
    repo_type="model",
)
print("Model uploaded.")
```

## Phase 5 - Post-publish quality checks

- [ ] Repo cards render correctly (YAML metadata + sections)
- [ ] Download + quick inference sanity check
- [ ] Dataset rows open and parse in the HF viewer
- [ ] Add linked references:
  - model card links dataset repo
  - dataset card links model repo
- [ ] Create release tags/changelog notes (`v0.1.0`)

## Release wording template

Use this wording in both cards:

> "This is a research preview release for structured logistics CX extraction. It is not production-certified. Use with human oversight for operational decisions."

