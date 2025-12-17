import re
from typing import Optional


def _strip_noise(val: str) -> str:
    # Remove currency symbols, NBSP, spaces, and leading non-numeric noise
    cleaned = val.replace("\u00a0", " ").strip()
    cleaned = re.sub(r"[^\d,.\-\+]", "", cleaned)
    cleaned = re.sub(r"^[^\d\-\+]+", "", cleaned)
    return cleaned


def parse_localized_number(
    s: str,
    *,
    decimal_hint: Optional[str] = None,
    thousands_hint: Optional[str] = None,
) -> Optional[float]:
    """
    Parses localized numeric strings (EU/US thousands & decimals). Returns None if not confident.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)

    val = str(s).strip()
    if not val:
        return None

    val = _strip_noise(val)
    if not val:
        return None

    dec_hint = decimal_hint or "."
    thou_hint = thousands_hint

    # Strong EU pattern: 1.234.567,89
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", val):
        candidate = val.replace(".", "").replace(",", ".")
        try:
            return float(candidate)
        except Exception:
            return None

    # Strong US pattern: 1,234,567.89
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", val):
        candidate = val.replace(",", "")
        try:
            return float(candidate)
        except Exception:
            return None

    # Multiple dots, no comma -> treat dots as thousands
    if val.count(".") > 1 and val.count(",") == 0:
        candidate = val.replace(".", "")
        try:
            return float(candidate)
        except Exception:
            return None

    # Single dot with 3-digit suffix => likely thousands separator
    if val.count(".") == 1 and val.count(",") == 0:
        parts = val.split(".")
        if len(parts) == 2 and len(parts[1]) == 3 and parts[0].isdigit() and parts[1].isdigit():
            candidate = "".join(parts)
            try:
                return float(candidate)
            except Exception:
                return None

    # Fallback using hints
    candidate = val
    if thou_hint:
        candidate = candidate.replace(thou_hint, "")
    # If still multiple separators, try removing thousands inferred from dec_hint
    if dec_hint and dec_hint in candidate:
        if dec_hint != ".":
            candidate = candidate.replace(dec_hint, ".")
    # Remove remaining thousands-like separators opposite to decimal
    other_sep = "," if dec_hint == "." else "."
    if candidate.count(dec_hint) == 1 and candidate.count(other_sep) >= 1:
        candidate = candidate.replace(other_sep, "")

    try:
        return float(candidate)
    except Exception:
        return None
