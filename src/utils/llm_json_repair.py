import json
import re
from typing import Any, Dict, List, Optional, Tuple


class JsonObjectParseError(ValueError):
    def __init__(self, message: str, trace: Dict[str, Any] | None = None):
        super().__init__(message)
        self.trace = trace or {}


def extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
                in_str = False
                escape = False
            continue
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def clean_json_payload(text: str) -> str:
    cleaned = re.sub(r"```json", "", str(text or ""), flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    return cleaned.strip()


def repair_common_json_damage(text: str) -> str:
    if not isinstance(text, str):
        return ""
    repaired = text.strip()
    if not repaired:
        return repaired

    repaired = clean_json_payload(repaired)
    extracted = extract_json_object(repaired)
    if extracted:
        repaired = extracted

    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

    repaired_lines: List[str] = []
    for raw_line in repaired.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        looks_like_json_line = stripped.startswith(("{", "}", "[", "]", "\"")) or ":" in stripped
        if (
            not looks_like_json_line
            and repaired_lines
            and repaired_lines[-1].rstrip().endswith('"')
            and not repaired_lines[-1].rstrip().endswith('",')
        ):
            prev = repaired_lines.pop().rstrip()
            if prev.endswith('"'):
                prev = prev[:-1]
            repaired_lines.append(f'{prev} {stripped.rstrip(",")}"')
            continue
        repaired_lines.append(line)
    if repaired_lines:
        repaired = "\n".join(repaired_lines)
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

    return repaired


def parse_json_object_with_repair(
    text: str,
    *,
    actor: str = "llm",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cleaned = clean_json_payload(text or "")
    repaired_cleaned = repair_common_json_damage(cleaned)
    repaired_raw = repair_common_json_damage(str(text or ""))
    candidates: List[Tuple[str, str | None]] = [
        ("cleaned", cleaned),
        ("extract_cleaned", extract_json_object(cleaned)),
        ("extract_raw", extract_json_object(str(text or ""))),
        ("repaired_cleaned", repaired_cleaned),
        ("extract_repaired_cleaned", extract_json_object(repaired_cleaned)),
        ("repaired_raw", repaired_raw),
        ("extract_repaired_raw", extract_json_object(repaired_raw)),
    ]

    parse_error: Exception | None = None
    seen: set[str] = set()
    attempts: List[Dict[str, Any]] = []
    chosen_step = ""

    for step, candidate in candidates:
        if not isinstance(candidate, str):
            continue
        blob = candidate.strip()
        if not blob or blob in seen:
            continue
        seen.add(blob)
        attempt_info: Dict[str, Any] = {"step": step, "chars": len(blob)}
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                attempt_info["ok"] = True
                attempts.append(attempt_info)
                chosen_step = step
                trace = {
                    "actor": str(actor or "llm"),
                    "used_repair": step not in {"cleaned", "extract_cleaned", "extract_raw"},
                    "chosen_step": chosen_step,
                    "attempts": attempts[-6:],
                }
                return parsed, trace
            parse_error = ValueError(f"{actor} JSON payload is not an object")
            attempt_info["ok"] = False
            attempt_info["error"] = "not_object"
        except Exception as err:
            parse_error = err
            attempt_info["ok"] = False
            attempt_info["error"] = str(err)[:220]
        attempts.append(attempt_info)

    trace = {
        "actor": str(actor or "llm"),
        "used_repair": False,
        "chosen_step": chosen_step or None,
        "attempts": attempts[-6:],
    }
    if parse_error:
        raise JsonObjectParseError(str(parse_error), trace)
    raise JsonObjectParseError(f"Empty {actor} JSON payload", trace)
