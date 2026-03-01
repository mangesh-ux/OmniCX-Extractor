"""
Local server for the finetuned logistics CX extractor.
Run on WSL to test inference and iterate on the system prompt.

  wsl
  cd /mnt/c/Users/mange/OneDrive/Desktop/OmniCX-Extractor
  python src/serve_inference.py

Then from WSL:
  curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d "{\"transcript\": \"Agent: Hi. Customer: Where is my order?\"}"
  # Optional: override system prompt for experiments
  curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d "{\"transcript\": \"...\", \"system_prompt\": \"You are a logistics analyst. Extract metrics.\"}"
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model once at startup (slow first request otherwise)
_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None:
        from inference import load_model
        print("Loading model (one-time)...")
        _model, _tokenizer = load_model()
        print("Model ready.")
    return _model, _tokenizer


@app.route("/")
@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "logistics-cx-extract"})


@app.route("/extract", methods=["POST"])
def extract():
    """POST JSON: { "transcript": "...", "system_prompt": "..." (optional) }."""
    try:
        body = request.get_json(force=True, silent=True) or {}
        transcript = body.get("transcript") or ""
        system_prompt = body.get("system_prompt")  # None = use default
        if not transcript.strip():
            return jsonify({"error": "Missing or empty 'transcript'"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    try:
        from inference import extract_with_finetuned
        model, tokenizer = get_model()
        result = extract_with_finetuned(
            transcript,
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            return_dict=True,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    import os
    os.chdir(REPO_ROOT)
    host = "0.0.0.0"
    port = 8000
    print(f"Serving at http://{host}:{port}")
    print("From Windows browser/curl: http://localhost:8000")
    print("POST /extract with JSON body: {\"transcript\": \"...\", \"system_prompt\": \"...\" (optional)}")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
