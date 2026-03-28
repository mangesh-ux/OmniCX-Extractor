"""
Publish prepared dataset and model artifacts to Hugging Face Hub.

Prerequisites:
  1) Run: python scripts/prepare_hf_release.py
  2) Login: hf auth login

Example:
  python scripts/publish_hf_artifacts.py \
    --dataset-repo your-username/omnicx-logistics-cx-dataset \
    --model-repo your-username/omnicx-qwen25-3b-lora
"""
from pathlib import Path
import argparse

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parent.parent
HF_RELEASE = REPO_ROOT / "hf_release"


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish OmniCX artifacts to Hugging Face Hub.")
    parser.add_argument("--dataset-repo", required=True, help="HF dataset repo id, e.g. user/omnicx-logistics-cx-dataset")
    parser.add_argument("--model-repo", required=True, help="HF model repo id, e.g. user/omnicx-qwen25-3b-lora")
    parser.add_argument("--private", action="store_true", help="Create repos as private")
    args = parser.parse_args()

    dataset_dir = HF_RELEASE / "dataset"
    model_dir = HF_RELEASE / "model"
    _require(dataset_dir)
    _require(model_dir)
    _require(dataset_dir / "README.md")
    _require(model_dir / "README.md")

    api = HfApi()

    # Dataset
    api.create_repo(repo_id=args.dataset_repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(dataset_dir), repo_id=args.dataset_repo, repo_type="dataset")
    print(f"Dataset published: https://huggingface.co/datasets/{args.dataset_repo}")

    # Model
    api.create_repo(repo_id=args.model_repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=args.model_repo, repo_type="model")
    print(f"Model published: https://huggingface.co/{args.model_repo}")


if __name__ == "__main__":
    main()
