
import re
import difflib
from typing import List, Dict, Any, Tuple

def normalize_colname(name: str) -> str:
    """
    Normalizes a column name for fuzzy matching:
    - Lowercase
    - Remove any non-alphanumeric character (keeps only a-z0-9)
    - Collapse validation
    
    Example: 'Customer ID' -> 'customerid', 'margin_net' -> 'marginnet'
    """
    if not isinstance(name, str):
        return str(name).lower()
    
    # Lowercase
    s = name.lower()
    # Remove any non-alphanumeric character
    s = re.sub(r'[^a-z0-9]', '', s)
    return s.strip()

def build_mapping(
    required_cols: List[str], 
    actual_cols: List[str], 
    allow_synthetic_margin: bool = True,
    *,
    enable_fuzzy: bool = True,
    fuzzy_cutoff: float = 0.86,
) -> Dict[str, Any]:
    """
    Builds a robust mapping from Required Columns -> Actual Columns.
    
    Protocol:
    1. Exact Match (Case-Insensitive)
    2. Exact Match (Normalized)
    3. Fuzzy Match (Difflib) when enabled
    4. Aliasing Check (One actual col cannot map to multiple required cols)
    5. Synthetic Logic (Margin/Profit defaults to 0.0 if missing)
    
    Returns:
        Dict: {
            "summary": { req_col: { "matched": actual, "method": "exact|fuzzy|synthetic", "score": 1.0 } },
            "mapped_df_cols": { actual_col: new_name }, # Renaming dict
            "missing": [list of missing required cols],
            "synthetic_created": [list of synthetic cols]
        }
    """
    mapping = {}
    used_actuals = set()
    missing = []
    synthetic = []
    
    # Normalize actuals for lookup (name -> norm) and (norm -> original)
    actual_map_norm = {normalize_colname(c): c for c in actual_cols}
    actual_map_lower = {c.lower(): c for c in actual_cols}
    
    for req in required_cols:
        req_norm = normalize_colname(req)
        match = None
        method = "none"
        score = 0.0
        
        # 1. Exact Match (Case Insensitive)
        if req.lower() in actual_map_lower:
            match = actual_map_lower[req.lower()]
            method = "exact_ci"
            score = 1.0
            
        # 2. Exact Match (Normalized)
        elif req_norm in actual_map_norm:
            match = actual_map_norm[req_norm]
            method = "exact_norm"
            score = 1.0
            
        # 3. Fuzzy Match
        elif enable_fuzzy:
            # Match against normalized keys
            candidates = difflib.get_close_matches(req_norm, actual_map_norm.keys(), n=1, cutoff=fuzzy_cutoff)
            if candidates:
                match_norm = candidates[0]
                match = actual_map_norm[match_norm]
                method = "fuzzy"
                # Simple score based on ratio
                score = difflib.SequenceMatcher(None, req_norm, match_norm).ratio()
        
        # 4. Aliasing & Final Assignment
        if match:
            if match in used_actuals:
                # ALIASING CONFLICT: This column is already mapped!
                # We do NOT assign it. Proceed to Missing/Synthetic check.
                match = None 
                # Record conflict in summary explicitly if needed, but the user requested:
                # "Si hay aliasing conflict (match ya usado), marca: {'matched': None, 'method': 'alias_conflict', ...} y añade req a missing."
                mapping[req] = {
                    "matched": None,
                    "method": "alias_conflict",
                    "score": 0.0,
                    "synthetic": False
                }
                missing.append(req)
            else:
                mapping[req] = {
                    "matched": match,
                    "method": method,
                    "score": round(score, 2),
                    "synthetic": False
                }
                used_actuals.add(match)
        
        # 5. Missing / Synthetic Handler
        if not match and req not in mapping: # Only process if not handled by alias conflict
            # Check if allowed synthetic
            is_margin_concept = "margin" in req.lower() or "profit" in req.lower() or "margen" in req.lower()
            if allow_synthetic_margin and is_margin_concept:
                mapping[req] = {
                    "matched": None,
                    "method": "synthetic",
                    "score": 0.0,
                    "synthetic": True
                }
                synthetic.append(req)
            else:
                mapping[req] = {
                    "matched": None,
                    "method": "missing",
                    "score": 0.0,
                    "synthetic": False
                }
                missing.append(req)

    # Build Output
    summary = mapping
    rename_dict = {}
    for req, details in mapping.items():
        if details["matched"]:
            rename_dict[details["matched"]] = req
            
    return {
        "summary": summary,
        "rename_mapping": rename_dict,
        "missing": missing,
        "synthetic": synthetic
    }
