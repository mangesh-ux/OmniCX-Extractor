"""
Keep only lines in eval_dataset.jsonl that parse (outer + inner JSON) and have
required keys. Normalizes double-escaped inner content. Overwrites with valid
lines only (backs up first).
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = REPO_ROOT / "data" / "eval" / "eval_dataset.jsonl"
REQUIRED_KEYS = ("behavioral_analytics", "operational_analytics", "diagnostic_reasoning")


def normalize_and_validate(line: str) -> str | None:
    """Parse line, normalize inner if double-escaped (\\"), validate. Return fixed line or None."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        msgs = obj.get("messages")
        if not msgs or len(msgs) < 3:
            return None
        raw = msgs[2].get("content", "")
        try:
            inner = json.loads(raw)
        except json.JSONDecodeError:
            inner_raw = raw.replace('\\"', '"')
            try:
                inner = json.loads(inner_raw)
            except json.JSONDecodeError:
                return None
            obj["messages"][2]["content"] = json.dumps(inner, ensure_ascii=False)
            line = json.dumps(obj, ensure_ascii=False)
        if not all(k in inner for k in REQUIRED_KEYS):
            return None
        return line
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def main():
    path = EVAL_PATH
    if not path.exists():
        print(f"Not found: {path}")
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    valid = []
    dropped = []
    for i, line in enumerate(lines, 1):
        out = normalize_and_validate(line)
        if out is not None:
            valid.append(out)
        else:
            dropped.append(i)
    if dropped:
        backup = path.with_suffix(path.suffix + ".bak")
        if backup.exists():
            backup.unlink()
        path.rename(backup)
        path.write_text("\n".join(valid) + "\n", encoding="utf-8")
        print(f"Kept {len(valid)} lines, removed {len(dropped)} erroneous (line numbers): {dropped}")
        print(f"Backup: {backup}")
    else:
        print(f"All {len(valid)} lines valid, no change.")


if __name__ == "__main__":
    main()
