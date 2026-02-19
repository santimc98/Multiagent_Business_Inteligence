import re
from typing import List, Tuple


_MAX_ERROR_TEXT_CHARS = 20000
_HINTS_HEADER = "REPAIR_HINTS (deterministic, no-autopatch):"


def _normalize_error_text(error_text: str) -> str:
    text = str(error_text or "")
    if len(text) > _MAX_ERROR_TEXT_CHARS:
        head = text[:12000]
        tail = text[-8000:]
        text = head + "\n...[TRUNCATED]...\n" + tail
    return text


def _match_invalid_categorical_feature_type(text: str) -> bool:
    invalid_type_patterns = [
        r"invalid type for cat_feature",
        r"invalid type for categorical feature",
    ]
    strong_semantic_combo = bool(
        re.search(
            r"(categorical|category|categoric).*must be.*(integer|int|string)",
            text,
            flags=re.IGNORECASE,
        )
        and re.search(r"(float|real number|0\.0|1\.0)", text, flags=re.IGNORECASE)
    )
    auxiliary_patterns = [
        r"(cat_features?|categorical features?).*(float|real number|0\.0|1\.0)",
    ]
    return bool(
        any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in invalid_type_patterns)
        or strong_semantic_combo
        or any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in auxiliary_patterns)
    )


def _match_nameerror_dialect_vars(text: str) -> bool:
    patterns = [
        r"nameerror:\s*name\s+'sep'\s+is\s+not\s+defined",
        r'nameerror:\s*name\s+"sep"\s+is\s+not\s+defined',
        r"nameerror:\s*name\s+'decimal'\s+is\s+not\s+defined",
        r'nameerror:\s*name\s+"decimal"\s+is\s+not\s+defined',
        r"nameerror:\s*name\s+'encoding'\s+is\s+not\s+defined",
        r'nameerror:\s*name\s+"encoding"\s+is\s+not\s+defined',
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _match_json_not_serializable(text: str) -> bool:
    return bool(
        re.search(
            r"object of type\s+['\"]?(bool_|int64|float64|timestamp)['\"]?\s+is not json serializable",
            text,
            flags=re.IGNORECASE,
        )
    )


def derive_repair_hints(error_text: str) -> List[str]:
    text = _normalize_error_text(error_text)
    prioritized_rules = [
        (
            _match_invalid_categorical_feature_type,
            "Tipo invalido en columnas categoricas: convierte las categorias a string o Int64 antes de entrenar; evita floats (0.0/1.0) como categorias y pasa las columnas categoricas como lo requiera tu stack (por nombre/indice/selector).",
        ),
        (
            _match_nameerror_dialect_vars,
            "Define sep/decimal/encoding desde output_dialect antes de usar read_csv/to_csv; no los uses sin inicializar.",
        ),
        (
            _match_json_not_serializable,
            "Serializacion JSON: usa json.dump(..., default=json_default) y/o convierte np.generic con .item() y Timestamps a str/ISO.",
        ),
    ]
    hints: List[str] = []
    for matcher, hint in prioritized_rules:
        if matcher(text) and hint not in hints:
            hints.append(hint)
        if len(hints) >= 2:
            break
    return hints


def append_repair_hints(feedback: str, error_text: str) -> Tuple[str, List[str]]:
    base_feedback = str(feedback or "").rstrip()
    hints = derive_repair_hints(error_text)
    if not hints:
        return base_feedback, []

    existing_lower = base_feedback.lower()
    missing_hints = [hint for hint in hints if hint.lower() not in existing_lower]
    if not missing_hints:
        return base_feedback, hints

    block = _HINTS_HEADER + "\n" + "\n".join([f"- {hint}" for hint in missing_hints])
    if base_feedback:
        return base_feedback + "\n\n" + block, hints
    return block, hints
