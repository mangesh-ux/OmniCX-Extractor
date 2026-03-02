"""
Fix eval_dataset.jsonl so each line is valid JSON.
Escapes double-quotes inside both user and assistant content values.
Reads data/eval/eval_dataset.jsonl, fixes, overwrites (with .bak backup).
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = REPO_ROOT / "data" / "eval" / "eval_dataset.jsonl"


def escape_json_string(s: str) -> str:
    """Escape backslash and double-quote for use inside a JSON string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def fix_line(line: str) -> str | None:
    """
    Fix one JSONL line by escaping " in user content and in assistant content.
    Extracts the three segments and rebuilds with proper escaping.
    """
    line = line.rstrip("\n\r")
    if not line.strip():
        return None

    # Find user content: between "role":"user","content":" and "},{"role":"assistant"
    # or "}, {"role": "assistant"
    user_markers = [
        ('"role":"user","content":"', '"},{"role":"assistant"'),
        ('"role": "user", "content": "', '"}, {"role": "assistant"'),
    ]
    user_start = user_end = None
    for start_m, end_m in user_markers:
        i = line.find(start_m)
        if i != -1:
            user_start = i + len(start_m)  # index of opening " of value
            j = line.find(end_m, user_start)
            if j != -1:
                user_end = j  # index of the " that closes user content (the " before })
                break
    if user_start is None or user_end is None:
        return None
    user_content = line[user_start + 1 : user_end]  # exclude opening and closing "
    user_escaped = escape_json_string(user_content)

    # Find assistant content: between "role":"assistant","content":" and "}]}"
    ast_markers = [
        '"role":"assistant","content":"',
        '"role": "assistant", "content": "',
    ]
    ast_start = None
    for m in ast_markers:
        i = line.find(m)
        if i != -1:
            ast_start = i + len(m)
            break
    if ast_start is None:
        return None
    end_marker = '"}]}'
    ast_end = line.rfind(end_marker)
    if ast_end == -1:
        return None
    assistant_content = line[ast_start + 1 : ast_end]
    # Normalize literal newlines to \n so inner JSON parses. Use placeholder so
    # escape_json_string doesn't double the backslash.
    _nl_ph = "\u0000\n\u0000"
    assistant_content = assistant_content.replace("\r\n", _nl_ph).replace("\n", _nl_ph).replace("\r", _nl_ph)
    # Fix already-double-escaped newline (\\\\\\\\n in extracted string -> \\n before escaping).
    assistant_content = assistant_content.replace("\\\\\\\\n", "\\\\n")
    assistant_escaped = escape_json_string(assistant_content)
    assistant_escaped = assistant_escaped.replace(_nl_ph, "\\n")

    # Rebuild: system (unchanged) + user (escaped) + assistant (escaped)
    # Line structure: {"messages":[ sys_block , user_block , ast_block ]}
    # We replace user_block content and ast_block content.
    sys_end = line.find('"role":"user"')
    if sys_end == -1:
        sys_end = line.find('"role": "user"')
    if sys_end == -1:
        return None
    # From start to end of system message (including comma before user)
    before_user = line[: user_start + 1]  # includes opening " of user content
    after_user_before_ast = line[user_end : ast_start + 1]  # "}, {"role":"assistant","content":"
    after_ast = line[ast_end:]

    return before_user + user_escaped + after_user_before_ast + assistant_escaped + after_ast


def main():
    path = EVAL_PATH
    if not path.exists():
        print(f"Not found: {path}")
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    fixed = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        out = fix_line(line)
        if out is None:
            print(f"Line {i+1}: could not parse structure, skipping")
            continue
        fixed.append(out)

    backup = path.with_suffix(path.suffix + ".bak")
    if backup.exists():
        backup.unlink()
    path.rename(backup)
    path.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    print(f"Fixed {len(fixed)} lines. Backup: {backup}")


if __name__ == "__main__":
    main()
