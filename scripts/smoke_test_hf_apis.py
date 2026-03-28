"""
Smoke-test Hugging Face model and dataset APIs for published OmniCX artifacts.

Checks:
1) model metadata resolves
2) dataset metadata resolves
3) required files exist in each repo
4) key small files are downloadable
5) datasets-server splits endpoint is reachable for dataset

Usage:
  python scripts/smoke_test_hf_apis.py
  python scripts/smoke_test_hf_apis.py --model-repo user/model --dataset-repo user/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_MODEL_REPO = "mangesh-ux/omnicx-logistics-cx-extractor-qwen25-3b-lora"
DEFAULT_DATASET_REPO = "mangesh-ux/logistics-cx-transcript-analysis-chatml"

REQUIRED_MODEL_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}

REQUIRED_DATASET_FILES = {
    "README.md",
    "train.jsonl",
    "test.jsonl",
}


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _check_datasets_server(dataset_repo: str) -> None:
    dataset_q = urllib.parse.quote(dataset_repo, safe="")
    url = f"https://datasets-server.huggingface.co/splits?dataset={dataset_q}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        body = resp.read().decode("utf-8")
    payload = json.loads(body)
    splits = {x["split"] for x in payload.get("splits", [])}
    missing = {"train", "test"} - splits
    if missing:
        raise RuntimeError(f"datasets-server missing splits: {sorted(missing)}")


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke-test Hugging Face model/dataset APIs.")
    p.add_argument("--model-repo", default=DEFAULT_MODEL_REPO, help="HF model repo id")
    p.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO, help="HF dataset repo id")
    args = p.parse_args()

    api = HfApi()
    failed = False

    # 1) Metadata resolution
    try:
        model_info = api.model_info(args.model_repo)
        _ok(f"Model metadata resolved: {model_info.id}")
    except Exception as e:
        failed = True
        _fail(f"Model metadata failed: {e}")

    try:
        dataset_info = api.dataset_info(args.dataset_repo)
        _ok(f"Dataset metadata resolved: {dataset_info.id}")
    except Exception as e:
        failed = True
        _fail(f"Dataset metadata failed: {e}")

    # 2) Required files exist
    try:
        model_files = set(api.list_repo_files(args.model_repo, repo_type="model"))
        missing = sorted(REQUIRED_MODEL_FILES - model_files)
        if missing:
            raise RuntimeError(f"missing model files: {missing}")
        _ok("Model required files present")
    except Exception as e:
        failed = True
        _fail(f"Model files check failed: {e}")

    try:
        dataset_files = set(api.list_repo_files(args.dataset_repo, repo_type="dataset"))
        missing = sorted(REQUIRED_DATASET_FILES - dataset_files)
        if missing:
            raise RuntimeError(f"missing dataset files: {missing}")
        _ok("Dataset required files present")
    except Exception as e:
        failed = True
        _fail(f"Dataset files check failed: {e}")

    # 3) Downloadability checks for small files
    try:
        hf_hub_download(repo_id=args.model_repo, filename="README.md", repo_type="model")
        hf_hub_download(repo_id=args.model_repo, filename="adapter_config.json", repo_type="model")
        _ok("Model small file downloads work")
    except Exception as e:
        failed = True
        _fail(f"Model download check failed: {e}")

    try:
        hf_hub_download(repo_id=args.dataset_repo, filename="README.md", repo_type="dataset")
        _ok("Dataset small file download works")
    except Exception as e:
        failed = True
        _fail(f"Dataset download check failed: {e}")

    # 4) Dataset viewer API (splits endpoint)
    try:
        _check_datasets_server(args.dataset_repo)
        _ok("datasets-server splits endpoint is healthy (train/test present)")
    except Exception as e:
        failed = True
        _fail(f"datasets-server check failed: {e}")

    if failed:
        print("\nHF API smoke test: FAILED")
        return 1

    print("\nHF API smoke test: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
