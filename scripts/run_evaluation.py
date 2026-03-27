"""
Evaluation script for the finetuned logistics CX extractor.

Loads an eval JSONL (ChatML format: messages with system, user, assistant).
Runs the model on each transcript, compares predictions to gold assistant JSON,
and writes a markdown report + optional JSON results.

Usage (from repo root):
  python scripts/run_evaluation.py --eval-file data/eval/eval_dataset.jsonl
  python scripts/run_evaluation.py --eval-file data/eval/eval_dataset.jsonl --report docs/training_logs/eval_report_iteration_001.md
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Keys we compare for accuracy (exact match)
BEHAVIORAL_KEYS = ["customer_intent", "customer_effort_score", "sentiment_trajectory", "rework_frequency"]
OPERATIONAL_KEYS = [
    "delivery_exception_type",
    "root_cause_category",
    "address_change_requested",
    "missed_delivery_explicitly_mentioned",
    "escalation_requested",
    "agent_explicitly_confirmed_resolution",
]
# unresolved_next_steps: we only check resolved vs not (N/A vs non-N/A)
DIAGNOSTIC_KEYS = []  # optional: recommended_routing_queue; skip for strictness (free text)

ALL_KEY_FIELDS = BEHAVIORAL_KEYS + OPERATIONAL_KEYS


def load_eval_jsonl(path: Path) -> list[dict]:
    """Load JSONL; each line = {"messages": [system, user, assistant]}."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                msgs = obj.get("messages") or []
                if len(msgs) < 3:
                    raise ValueError(f"Line {i}: expected 3 messages, got {len(msgs)}")
                rows.append({"messages": msgs, "line_num": i})
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {i}: invalid JSON: {e}") from e
    return rows


def get_transcript_and_gold(row: dict) -> tuple[str, dict]:
    """Extract user transcript and parsed gold assistant JSON."""
    msgs = row["messages"]
    transcript = (msgs[1].get("content") or "").strip()
    raw_assistant = msgs[2].get("content")
    if isinstance(raw_assistant, dict):
        gold = raw_assistant
    else:
        raw_assistant = (raw_assistant or "").strip()
        gold = json.loads(raw_assistant)
    return transcript, gold


def normalize_resolved(s: str) -> str:
    """Normalize unresolved_next_steps for comparison: N/A vs non-N/A."""
    if not s or not isinstance(s, str):
        return "N/A"
    t = s.strip().upper()
    if t == "N/A" or t == "NA":
        return "N/A"
    return "PENDING"


def _get_behavioral(d: dict) -> dict:
    return (d.get("behavioral_analytics") or {})

def _get_operational(d: dict) -> dict:
    return (d.get("operational_analytics") or {})


def compare_one(gold: dict, pred: dict) -> dict:
    """Per-field match and strict (all key fields). Returns dict of field -> bool + 'strict'."""
    out = {}
    g_beh, g_op = _get_behavioral(gold), _get_operational(gold)
    p_beh, p_op = _get_behavioral(pred), _get_operational(pred)
    for key in BEHAVIORAL_KEYS:
        out[key] = (g_beh.get(key) == p_beh.get(key))
    for key in OPERATIONAL_KEYS:
        out[key] = (g_op.get(key) == p_op.get(key))
    go = g_op.get("unresolved_next_steps", "")
    po = p_op.get("unresolved_next_steps", "")
    out["unresolved_next_steps"] = normalize_resolved(go) == normalize_resolved(po)
    out["strict"] = all(out[k] for k in ALL_KEY_FIELDS) and out["unresolved_next_steps"]
    return out


def run_evaluation(eval_path: Path, model_path: Path = None, report_path: Path = None, results_json_path: Path = None, outputs_path: Path = None) -> dict:
    """Load eval JSONL, run model, compare, write report. Returns summary stats."""
    from inference import load_model, extract_with_finetuned

    eval_path = Path(eval_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    rows = load_eval_jsonl(eval_path)
    if not rows:
        raise ValueError("No valid rows in eval file")

    model_path = model_path or REPO_ROOT / "models" / "qwen-logistics-lora"
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_model(model_path=model_path)
    print(f"Running inference on {len(rows)} examples...")

    results = []
    for row in rows:
        transcript, gold = get_transcript_and_gold(row)
        t0 = time.perf_counter()
        try:
            pred = extract_with_finetuned(transcript, model=model, tokenizer=tokenizer, return_dict=True)
            latency_sec = time.perf_counter() - t0
        except Exception as e:
            latency_sec = time.perf_counter() - t0
            results.append({
                "line_num": row["line_num"],
                "gold": gold,
                "pred": None,
                "error": str(e),
                "match": None,
                "latency_sec": latency_sec,
            })
            continue
        match = compare_one(gold, pred)
        results.append({
            "line_num": row["line_num"],
            "gold": gold,
            "pred": pred,
            "match": match,
            "latency_sec": latency_sec,
        })

    # Aggregate
    key_fields = ALL_KEY_FIELDS + ["unresolved_next_steps"]
    correct = {k: 0 for k in key_fields}
    correct["strict"] = 0
    valid = 0
    for r in results:
        if r["match"] is None:
            continue
        valid += 1
        for k in key_fields:
            if r["match"].get(k):
                correct[k] += 1
        if r["match"].get("strict"):
            correct["strict"] += 1

    total = len(results)
    latencies = [r["latency_sec"] for r in results]
    summary = {
        "total": total,
        "valid": valid,
        "errors": total - valid,
        "per_field_accuracy": {k: (correct[k] / valid if valid else 0) for k in key_fields},
        "strict_accuracy": correct["strict"] / valid if valid else 0,
        "latency_sec": {
            "mean": sum(latencies) / len(latencies) if latencies else 0,
            "min": min(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0,
            "total": sum(latencies),
        },
    }
    # Log latency to console
    lat = summary["latency_sec"]
    print(f"Latency: mean={lat['mean']:.2f}s, min={lat['min']:.2f}s, max={lat['max']:.2f}s, total={lat['total']:.2f}s")

    # Report path
    report_path = report_path or REPO_ROOT / "docs" / "training_logs" / "eval_report_iteration_001.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Write markdown report
    lines = [
        "# Evaluation Report — Iteration 001",
        "",
        f"**Eval file:** `{eval_path.name}`  ",
        f"**Total examples:** {total}  ",
        f"**Valid (no runtime error):** {valid}  ",
        f"**Errors:** {total - valid}  ",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|--------|",
        f"| Strict accuracy (all key fields match) | {summary['strict_accuracy']:.1%} ({correct['strict']}/{valid}) |",
        f"| Latency (mean) | {summary['latency_sec']['mean']:.2f}s |",
        f"| Latency (min / max) | {summary['latency_sec']['min']:.2f}s / {summary['latency_sec']['max']:.2f}s |",
        f"| Latency (total) | {summary['latency_sec']['total']:.2f}s |",
        "",
        "## Per-field accuracy",
        "",
        "| Field | Correct | Accuracy |",
        "|-------|---------|----------|",
    ]
    for k in key_fields:
        pct = summary["per_field_accuracy"][k]
        lines.append(f"| {k} | {correct[k]}/{valid} | {pct:.1%} |")
    lines.extend(["", "## Failures (strict)"])

    failures = [r for r in results if r.get("match") and not r["match"].get("strict")]
    if not failures:
        lines.append("")
        lines.append("None.")
    else:
        for r in failures:
            ln = r["line_num"]
            missed = [k for k in key_fields if not r["match"].get(k)]
            lines.append(f"- **Line {ln}**: missed {', '.join(missed)}")

    if any(r.get("error") for r in results):
        lines.extend(["", "## Runtime errors"])
        for r in results:
            if r.get("error"):
                lines.append(f"- Line {r['line_num']}: {r['error'][:200]}")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")

    if results_json_path:
        results_json_path = Path(results_json_path)
        results_json_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize for JSON (no numpy); include full pred and latency
        out_json = {
            "summary": summary,
            "results": [
                {
                    "line_num": r["line_num"],
                    "strict_match": r.get("match", {}).get("strict") if r.get("match") else None,
                    "match": r.get("match"),
                    "pred": r.get("pred"),
                    "gold": r.get("gold"),
                    "error": r.get("error"),
                    "latency_sec": r.get("latency_sec"),
                }
                for r in results
            ],
        }
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2)
        print(f"Results JSON written to {results_json_path}")

    # Save model outputs to a JSONL (one line per example: pred + latency)
    if outputs_path:
        outputs_path = Path(outputs_path)
        outputs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(outputs_path, "w", encoding="utf-8") as f:
            for r in results:
                rec = {
                    "line_num": r["line_num"],
                    "pred": r.get("pred"),
                    "gold": r.get("gold"),
                    "latency_sec": r.get("latency_sec"),
                    "strict_match": r.get("match", {}).get("strict") if r.get("match") else None,
                    "error": r.get("error"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Model outputs written to {outputs_path}")

    return summary


def main():
    p = argparse.ArgumentParser(description="Evaluate finetuned logistics CX model on a ChatML JSONL.")
    p.add_argument("--eval-file", type=Path, default=REPO_ROOT / "data" / "eval" / "eval_dataset.jsonl", help="Path to eval JSONL")
    p.add_argument("--report", type=Path, default=None, help="Output markdown report path")
    p.add_argument("--results-json", type=Path, default=None, help="Optional: output detailed results JSON (includes pred + latency)")
    p.add_argument("--outputs", type=Path, default=REPO_ROOT / "docs" / "training_logs" / "eval_outputs_iteration_001.jsonl", help="Save model outputs to JSONL (pred, gold, latency per line)")
    p.add_argument("--model-path", type=Path, default=None, help="Path to LoRA adapter (default: models/qwen-logistics-lora)")
    args = p.parse_args()
    run_evaluation(
        eval_path=args.eval_file,
        model_path=args.model_path,
        report_path=args.report,
        results_json_path=args.results_json,
        outputs_path=args.outputs,
    )


if __name__ == "__main__":
    main()
