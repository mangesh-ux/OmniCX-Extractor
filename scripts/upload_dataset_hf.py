"""
Upload prepared dataset artifacts to a Hugging Face dataset repo.

Prerequisites:
  hf auth login
  python scripts/prepare_hf_release.py

Example:
  python scripts/upload_dataset_hf.py --repo mangesh-ux/logistics-cx-transcript-analysis-chatml
"""
from pathlib import Path
import argparse

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "hf_release" / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload hf_release/dataset to a HF dataset repo.")
    parser.add_argument("--repo", required=True, help="Dataset repo id, e.g. user/repo-name")
    args = parser.parse_args()

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")
    if not (DATASET_DIR / "README.md").exists():
        raise FileNotFoundError(f"Missing dataset card: {DATASET_DIR / 'README.md'}")

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(DATASET_DIR),
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"Dataset uploaded to: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
