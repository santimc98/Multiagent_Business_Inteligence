from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Tuple


def _garble_once(text: str, codec: str) -> str | None:
    try:
        return text.encode("utf-8", errors="strict").decode(codec, errors="strict")
    except Exception:
        return None


def _build_suspect_snippets() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    # Common non-ASCII characters seen in multilingual business reports.
    sample_chars = (
        "\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1"
        "\u00c1\u00c9\u00cd\u00d3\u00da\u00d1"
        "\u00fc\u00dc\u00bf\u00a1"
        "\u2018\u2019\u201c\u201d\u2013\u2014\u2026\u2022\u2122"
        "\u20ac\u00a9\u00ae"
    )
    single = set()
    double = set()
    for char in sample_chars:
        for codec in ("latin-1", "cp1252"):
            once = _garble_once(char, codec)
            if once and once != char:
                single.add(once)
                twice = _garble_once(once, codec)
                if twice and twice != once:
                    double.add(twice)
    single_tokens = tuple(sorted(single, key=lambda token: (-len(token), token)))
    double_tokens = tuple(sorted(double, key=lambda token: (-len(token), token)))
    return single_tokens, double_tokens


_SUSPECT_SINGLE_SNIPPETS, _SUSPECT_DOUBLE_SNIPPETS = _build_suspect_snippets()

# Character-level clues that strongly indicate broken transcoding.
_C1_CONTROL_RE = re.compile(r"[\u0080-\u009F]")
_SUSPICIOUS_PAIR_RE = re.compile(r"[\u00C2\u00C3][\u00A0-\u00BF\u0080-\u009F]")


def normalize_unicode_nfc(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFC", text)


def mojibake_score(text: str) -> int:
    if not text:
        return 0
    score = 0
    for token in _SUSPECT_DOUBLE_SNIPPETS:
        score += text.count(token) * 7
    for token in _SUSPECT_SINGLE_SNIPPETS:
        score += text.count(token) * 4

    score += len(_C1_CONTROL_RE.findall(text)) * 5
    score += text.count("\u00AD") * 5  # soft hyphen frequently appears in broken "\u00c3\u00ad"
    score += text.count("\uFFFD") * 8
    score += len(_SUSPICIOUS_PAIR_RE.findall(text)) * 3
    return score


def has_mojibake(text: str) -> bool:
    return mojibake_score(text or "") > 0


def _transcode_candidate(text: str, source_encoding: str) -> str | None:
    try:
        repaired = text.encode(source_encoding, errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return None
    return normalize_unicode_nfc(repaired)


def repair_mojibake_if_needed(text: str, max_rounds: int = 3, min_improvement: int = 1) -> str:
    candidate = normalize_unicode_nfc(text or "")
    current_score = mojibake_score(candidate)
    if current_score <= 0:
        return candidate

    best = candidate
    best_score = current_score
    for _ in range(max_rounds):
        improved = False
        for source_encoding in ("cp1252", "latin-1"):
            repaired = _transcode_candidate(best, source_encoding)
            if repaired is None or repaired == best:
                continue
            repaired_score = mojibake_score(repaired)
            if repaired_score <= max(0, best_score - min_improvement):
                best = repaired
                best_score = repaired_score
                improved = True
        if not improved:
            break
    return best


def sanitize_text(text: str) -> str:
    normalized = normalize_unicode_nfc(text or "")
    return repair_mojibake_if_needed(normalized)


def _empty_stats() -> Dict[str, int]:
    return {"strings_seen": 0, "strings_changed": 0, "mojibake_hits": 0}


def _sanitize_payload_internal(value: Any) -> Tuple[Any, Dict[str, int]]:
    stats = _empty_stats()
    if isinstance(value, str):
        stats["strings_seen"] = 1
        original = normalize_unicode_nfc(value)
        if has_mojibake(original):
            stats["mojibake_hits"] = 1
        repaired = sanitize_text(original)
        if repaired != original:
            stats["strings_changed"] = 1
        return repaired, stats
    if isinstance(value, (bytes, bytearray)):
        try:
            decoded = value.decode("utf-8", errors="replace")
        except Exception:
            decoded = str(value)
        return _sanitize_payload_internal(decoded)
    if isinstance(value, list):
        out = []
        agg = _empty_stats()
        for item in value:
            fixed_item, item_stats = _sanitize_payload_internal(item)
            out.append(fixed_item)
            for key, val in item_stats.items():
                agg[key] = agg.get(key, 0) + int(val)
        return out, agg
    if isinstance(value, tuple):
        out = []
        agg = _empty_stats()
        for item in value:
            fixed_item, item_stats = _sanitize_payload_internal(item)
            out.append(fixed_item)
            for key, val in item_stats.items():
                agg[key] = agg.get(key, 0) + int(val)
        return tuple(out), agg
    if isinstance(value, set):
        out = set()
        agg = _empty_stats()
        for item in value:
            fixed_item, item_stats = _sanitize_payload_internal(item)
            out.add(fixed_item)
            for key, val in item_stats.items():
                agg[key] = agg.get(key, 0) + int(val)
        return out, agg
    if isinstance(value, dict):
        out = {}
        agg = _empty_stats()
        for key, val in value.items():
            if isinstance(key, str):
                fixed_key, key_stats = _sanitize_payload_internal(key)
            else:
                fixed_key, key_stats = key, _empty_stats()
            fixed_val, val_stats = _sanitize_payload_internal(val)
            out[fixed_key] = fixed_val
            for key_name, key_val in key_stats.items():
                agg[key_name] = agg.get(key_name, 0) + int(key_val)
            for key_name, key_val in val_stats.items():
                agg[key_name] = agg.get(key_name, 0) + int(key_val)
        return out, agg
    return value, stats


def sanitize_text_payload(value: Any) -> Any:
    sanitized, _ = _sanitize_payload_internal(value)
    return sanitized


def sanitize_text_payload_with_stats(value: Any) -> Tuple[Any, Dict[str, int]]:
    return _sanitize_payload_internal(value)
