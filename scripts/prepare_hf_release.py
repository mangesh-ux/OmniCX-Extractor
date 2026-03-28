"""
Prepare Hugging Face release folders for dataset and model artifacts.

Creates:
  hf_release/dataset/
  hf_release/model/

Copies available files from repo and prints blockers (e.g., missing model weights).
"""
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parent.parent
HF_RELEASE = REPO_ROOT / "hf_release"
DATASET_OUT = HF_RELEASE / "dataset"
MODEL_OUT = HF_RELEASE / "model"


def _copy_if_exists(src: Path, dst: Path, warnings: list[str]) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        warnings.append(f"Missing: {src}")


def main() -> None:
    warnings: list[str] = []
    HF_RELEASE.mkdir(parents=True, exist_ok=True)
    DATASET_OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    # Dataset card + files
    _copy_if_exists(REPO_ROOT / "docs" / "hf" / "DATASET_CARD.md", DATASET_OUT / "README.md", warnings)
    _copy_if_exists(REPO_ROOT / "data" / "processed" / "golden_training_dataset.jsonl", DATASET_OUT / "train.jsonl", warnings)
    _copy_if_exists(REPO_ROOT / "data" / "eval" / "eval_dataset.jsonl", DATASET_OUT / "test.jsonl", warnings)

    # Model card + adapter/tokenizer artifacts
    _copy_if_exists(REPO_ROOT / "docs" / "hf" / "MODEL_CARD.md", MODEL_OUT / "README.md", warnings)
    _copy_if_exists(REPO_ROOT / "models" / "qwen-logistics-lora" / "adapter_config.json", MODEL_OUT / "adapter_config.json", warnings)
    _copy_if_exists(REPO_ROOT / "models" / "qwen-logistics-lora" / "adapter_model.safetensors", MODEL_OUT / "adapter_model.safetensors", warnings)
    _copy_if_exists(REPO_ROOT / "models" / "qwen-logistics-lora" / "tokenizer.json", MODEL_OUT / "tokenizer.json", warnings)
    _copy_if_exists(REPO_ROOT / "models" / "qwen-logistics-lora" / "tokenizer_config.json", MODEL_OUT / "tokenizer_config.json", warnings)
    _copy_if_exists(REPO_ROOT / "models" / "qwen-logistics-lora" / "chat_template.jinja", MODEL_OUT / "chat_template.jinja", warnings)
    _copy_if_exists(
        REPO_ROOT / "docs" / "training_logs" / "eval_report_iteration_001.md",
        MODEL_OUT / "eval_report_iteration_001.md",
        warnings,
    )
    _copy_if_exists(
        REPO_ROOT / "docs" / "training_logs" / "eval_outputs_iteration_001.jsonl",
        MODEL_OUT / "eval_outputs_iteration_001.jsonl",
        warnings,
    )

    print(f"Prepared HF release folders at: {HF_RELEASE}")
    if warnings:
        print("\nBlockers / missing files:")
        for w in warnings:
            print(f"- {w}")
    else:
        print("\nAll expected files copied successfully.")


if __name__ == "__main__":
    main()
