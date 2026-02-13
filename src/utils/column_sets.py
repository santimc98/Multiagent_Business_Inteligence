import re
from typing import Any, Dict, List

_NUMERIC_SUFFIX_RE = re.compile(r"^(.*?)(\d+)$")
_EXPLICIT_ROLE_ALLOWLIST = {"target_candidate", "id_like", "split_candidate", "constant_like"}


def _normalize_columns(columns: List[Any]) -> List[str]:
    return [str(col) for col in columns if col is not None and str(col).strip()]


def _collect_explicit_columns(columns: List[str], roles: Dict[str, str] | None) -> List[str]:
    if not isinstance(roles, dict) or not roles:
        return []
    explicit: List[str] = []
    seen = set()
    for col in columns:
        if roles.get(col) in _EXPLICIT_ROLE_ALLOWLIST and col not in seen:
            seen.add(col)
            explicit.append(col)
    return explicit


def _match_selector_columns(
    columns: List[str],
    selector: Dict[str, Any],
    *,
    excluded: set[str] | None = None,
) -> List[str]:
    selector = selector if isinstance(selector, dict) else {}
    selector_type = str(selector.get("type") or "").strip().lower()
    excluded_set = excluded if isinstance(excluded, set) else set()
    if not columns or not selector_type:
        return []

    matched: List[str] = []
    if selector_type == "prefix_numeric_range":
        prefix = str(selector.get("prefix") or "")
        start = selector.get("start")
        end = selector.get("end")
        if prefix and isinstance(start, int) and isinstance(end, int):
            lo = min(start, end)
            hi = max(start, end)
            for col in columns:
                if col in excluded_set:
                    continue
                match = _NUMERIC_SUFFIX_RE.match(col)
                if not match:
                    continue
                if match.group(1) != prefix:
                    continue
                idx = int(match.group(2))
                if lo <= idx <= hi:
                    matched.append(col)
    elif selector_type == "regex":
        pattern = selector.get("pattern")
        if pattern:
            try:
                regex = re.compile(str(pattern))
            except re.error:
                return []
            matched = [col for col in columns if col not in excluded_set and regex.match(col)]
    elif selector_type == "prefix":
        token = str(selector.get("value") or selector.get("prefix") or "")
        if token:
            matched = [col for col in columns if col not in excluded_set and col.startswith(token)]
    elif selector_type == "all_numeric_except":
        excluded_cols = set(_normalize_columns(selector.get("except_columns") or []))
        excluded_cols |= excluded_set
        matched = [col for col in columns if col not in excluded_cols]
    elif selector_type == "all_columns_except":
        excluded_cols = set(_normalize_columns(selector.get("except_columns") or []))
        excluded_cols |= excluded_set
        matched = [col for col in columns if col not in excluded_cols]
    elif selector_type == "list":
        listed = _normalize_columns(selector.get("columns") or [])
        for col in listed:
            if col in excluded_set:
                continue
            if col in columns and col not in matched:
                matched.append(col)
    return matched


def build_column_sets(
    columns: List[str],
    roles: Dict[str, str] | None = None,
    max_listed: int = 50,
) -> Dict[str, Any]:
    cols = _normalize_columns(columns or [])
    explicit_columns = _collect_explicit_columns(cols, roles)
    explicit_set = set(explicit_columns)
    remaining = [col for col in cols if col not in explicit_set]

    groups: Dict[str, List[tuple[int, str]]] = {}
    for col in remaining:
        match = _NUMERIC_SUFFIX_RE.match(col)
        if not match:
            continue
        prefix = match.group(1)
        if not prefix:
            continue
        idx = int(match.group(2))
        groups.setdefault(prefix, []).append((idx, col))

    sets: List[Dict[str, Any]] = []
    covered: set[str] = set()
    set_index = 1

    for prefix in sorted(groups.keys()):
        items = groups[prefix]
        if len(items) < 50:
            continue
        indices = sorted({idx for idx, _ in items})
        start = min(indices)
        end = max(indices)
        span = max(end - start + 1, 1)
        coverage = len(indices) / span
        matched = [col for _, col in items]

        if coverage >= 0.95:
            selector = {"type": "prefix_numeric_range", "prefix": prefix, "start": int(start), "end": int(end)}
        else:
            selector = {"type": "regex", "pattern": r"^" + re.escape(prefix) + r"\d+$"}

        if len(matched) >= 50:
            sets.append({"name": f"SET_{set_index}", "selector": selector, "count": len(matched)})
            covered.update(matched)
            set_index += 1

    if not sets and len(cols) >= 700:
        selector = {"type": "all_columns_except", "except_columns": explicit_columns}
        sets.append({"name": "SET_1", "selector": selector, "count": len(remaining)})
        covered.update(remaining)

    leftovers = [col for col in remaining if col not in covered]
    leftovers_sample = {
        "columns": leftovers[:max_listed],
        "total_leftovers": len(leftovers),
    }

    return {
        "explicit_columns": explicit_columns,
        "sets": sets,
        "leftovers_sample": leftovers_sample,
        "leftovers_count": len(leftovers),
    }


def expand_column_sets(columns: List[str], sets_spec: Dict[str, Any]) -> Dict[str, Any]:
    cols = _normalize_columns(columns or [])
    explicit_columns = _normalize_columns(sets_spec.get("explicit_columns") or [])
    explicit_set = set(explicit_columns)
    sets = sets_spec.get("sets") if isinstance(sets_spec, dict) else []
    expanded: List[str] = []
    expanded_seen = set()
    debug: Dict[str, int] = {}

    for entry in sets if isinstance(sets, list) else []:
        name = entry.get("name") or "SET"
        selector = entry.get("selector") if isinstance(entry, dict) else {}
        if not isinstance(selector, dict):
            selector = {}
        matched = _match_selector_columns(cols, selector, excluded=explicit_set)

        for col in matched:
            if col in expanded_seen or col in explicit_set:
                continue
            expanded_seen.add(col)
            expanded.append(col)
        debug[name] = len(matched)

    return {
        "expanded_feature_columns": expanded,
        "debug": debug,
    }


def build_column_manifest(
    columns: List[str],
    *,
    column_sets: Dict[str, Any] | None = None,
    roles: Dict[str, str] | None = None,
    min_family_size: int = 10,
) -> Dict[str, Any]:
    """
    Build a compact schema manifest that is stable for wide datasets.

    Notes:
    - This manifest is structural (header-based). It does not infer business roles
      beyond anchor hints already provided by upstream agents/roles.
    - It complements (does not replace) contract/view artifacts.
    """
    cols = _normalize_columns(columns or [])
    sets_spec = column_sets if isinstance(column_sets, dict) else {}
    explicit_from_sets = _normalize_columns(sets_spec.get("explicit_columns") or [])
    anchors = explicit_from_sets or _collect_explicit_columns(cols, roles)
    anchor_set = set(anchors)

    families: List[Dict[str, Any]] = []
    sets_raw = sets_spec.get("sets") if isinstance(sets_spec.get("sets"), list) else []
    next_family_id = 1
    for entry in sets_raw:
        if not isinstance(entry, dict):
            continue
        selector = entry.get("selector")
        if not isinstance(selector, dict):
            continue
        matched = _match_selector_columns(cols, selector, excluded=anchor_set)
        if len(matched) < int(max(1, min_family_size)):
            continue
        family_name = str(entry.get("name") or "").strip()
        if not family_name:
            family_name = f"family_{next_family_id}"
            next_family_id += 1
        family = {
            "family_id": family_name,
            "selector": selector,
            "count": len(matched),
            "examples": [matched[0], matched[len(matched) // 2], matched[-1]],
            "role": "feature",
        }
        families.append(family)

    schema_mode = "wide" if len(cols) > 200 and bool(families) else "normal"
    covered = set()
    for family in families:
        selector = family.get("selector")
        covered.update(_match_selector_columns(cols, selector if isinstance(selector, dict) else {}, excluded=anchor_set))
    leftovers = [col for col in cols if col not in anchor_set and col not in covered]

    return {
        "schema_mode": schema_mode,
        "total_columns": len(cols),
        "anchors": anchors,
        "families": families,
        "anchor_count": len(anchors),
        "family_count": len(families),
        "covered_family_columns": len(covered),
        "leftovers_count": len(leftovers),
        "leftovers_sample": leftovers[:40],
    }


def summarize_column_manifest(column_manifest: Dict[str, Any]) -> str:
    if not isinstance(column_manifest, dict) or not column_manifest:
        return ""
    mode = str(column_manifest.get("schema_mode") or "unknown")
    total = column_manifest.get("total_columns")
    anchors = column_manifest.get("anchors") if isinstance(column_manifest.get("anchors"), list) else []
    families = column_manifest.get("families") if isinstance(column_manifest.get("families"), list) else []
    lines = ["COLUMN_MANIFEST_SUMMARY:"]
    lines.append(f"- schema_mode: {mode}")
    if isinstance(total, int):
        lines.append(f"- total_columns: {total}")
    lines.append(f"- anchors_count: {len(anchors)}")
    if anchors:
        lines.append(f"- anchors_sample: {anchors[:10]}")
    lines.append(f"- families_count: {len(families)}")
    if families:
        family_bits: List[str] = []
        for fam in families[:6]:
            if not isinstance(fam, dict):
                continue
            family_id = str(fam.get("family_id") or "family")
            selector = fam.get("selector") if isinstance(fam.get("selector"), dict) else {}
            sel_type = str(selector.get("type") or "unknown")
            count = fam.get("count")
            if sel_type == "prefix_numeric_range":
                detail = f"{selector.get('prefix')}{selector.get('start')}..{selector.get('end')}"
            elif sel_type == "regex":
                detail = str(selector.get("pattern") or "")
            else:
                detail = sel_type
            count_txt = str(count) if isinstance(count, int) else "?"
            family_bits.append(f"{family_id}({detail}, count={count_txt})")
        if family_bits:
            lines.append(f"- families: {family_bits}")
    leftovers_count = column_manifest.get("leftovers_count")
    if isinstance(leftovers_count, int) and leftovers_count > 0:
        lines.append(f"- leftovers_count: {leftovers_count}")
    return "\n".join(lines)


def summarize_column_sets(column_sets: Dict[str, Any], max_sets: int = 5) -> str:
    if not isinstance(column_sets, dict) or not column_sets:
        return ""
    explicit = column_sets.get("explicit_columns") or []
    sets = column_sets.get("sets") or []
    leftovers = column_sets.get("leftovers_sample") or {}
    lines = ["COLUMN_SETS_SUMMARY:"]
    lines.append(f"- explicit_columns_count: {len(explicit)}")
    if sets:
        set_summaries = []
        for entry in sets[:max_sets]:
            selector = entry.get("selector") if isinstance(entry, dict) else {}
            selector_type = selector.get("type") if isinstance(selector, dict) else "unknown"
            count = entry.get("count")
            name = entry.get("name") or "SET"
            if selector_type == "prefix_numeric_range":
                prefix = selector.get("prefix")
                start = selector.get("start")
                end = selector.get("end")
                detail = f"{prefix}{start}-{end}"
            elif selector_type == "regex":
                detail = selector.get("pattern")
            else:
                detail = selector_type
            count_text = count if isinstance(count, int) else "unknown"
            set_summaries.append(f"{name}({detail}, count={count_text})")
        lines.append(f"- sets: {set_summaries}")
    else:
        lines.append("- sets: none")
    total_leftovers = leftovers.get("total_leftovers")
    if isinstance(total_leftovers, int) and total_leftovers > 0:
        lines.append(f"- leftovers_total: {total_leftovers}")
    return "\n".join(lines)
