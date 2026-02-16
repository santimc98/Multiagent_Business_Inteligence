from __future__ import annotations

import re
from typing import Any, Dict, List


_GATE_LINE_RE = re.compile(
    r"^\s*Gate\s+([A-Za-z0-9]+)\s*:\s*(PASS|FAIL)\b(.*)$",
    flags=re.IGNORECASE,
)


def _extract_preflight_block(text: str, max_lines: int = 80) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    marker = "pre_flight_gates"
    lower = text.lower()
    marker_idx = lower.rfind(marker)
    if marker_idx < 0:
        return ""

    tail = text[marker_idx:]
    lines = tail.splitlines()
    collected: List[str] = []
    marker_seen = False
    gate_lines = 0

    for raw_line in lines[:max_lines]:
        line = str(raw_line)
        stripped = line.strip()
        lowered = stripped.lower()

        if not marker_seen:
            if marker in lowered:
                marker_seen = True
                collected.append(line)
            continue

        if re.match(r"^\s*gate\s+[a-z0-9]+\s*:", line, flags=re.IGNORECASE):
            collected.append(line)
            gate_lines += 1
            continue

        if gate_lines == 0:
            if not stripped or set(stripped) <= {"-", "=", ":"}:
                collected.append(line)
                continue
            collected.append(line)
            continue

        # Stop once another strong section likely starts.
        if stripped.startswith("{") or stripped.startswith("["):
            break
        if lowered.startswith("contract_execution_map"):
            break
        if stripped.isupper() and ":" not in stripped and not lowered.startswith("gate "):
            break
        collected.append(line)

    return "\n".join(collected).strip()


def extract_preflight_gate_failures(text: str) -> List[Dict[str, str]]:
    """
    Parse PRE_FLIGHT_GATES output and return FAIL entries only.
    """
    block = _extract_preflight_block(text)
    if not block:
        return []

    failures: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in block.splitlines():
        match = _GATE_LINE_RE.match(line)
        if not match:
            continue
        gate = str(match.group(1) or "").strip().upper()
        status = str(match.group(2) or "").strip().upper()
        detail = str(match.group(3) or "").strip()
        detail = re.sub(r"^[\s\-:]+", "", detail)
        if status != "FAIL" or not gate:
            continue
        key = (gate, detail.lower())
        if key in seen:
            continue
        seen.add(key)
        failures.append(
            {
                "gate": gate,
                "status": status,
                "detail": detail,
            }
        )
    return failures


def extract_preflight_gate_tail(text: str, max_chars: int = 700) -> str:
    """
    Return a compact raw PRE_FLIGHT_GATES block tail for audit context.
    """
    block = _extract_preflight_block(text)
    if not block:
        return ""
    cleaned = block.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    keep = max(1, max_chars - 4)
    return "...\n" + cleaned[-keep:]
