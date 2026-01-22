
from typing import Dict, Any, List

def compact_data_profile_for_llm(profile: Dict[str, Any], max_cols: int = 60) -> Dict[str, Any]:
    """
    Compact the data profile for LLM consumption.
    Retains critical decision-making facts while reducing token usage.
    """
    if not isinstance(profile, dict):
        return {}

    compact = {}
    
    # 1. Basic Stats (Critical)
    compact["basic_stats"] = profile.get("basic_stats", {})
    
    # 2. Outcome Analysis (Critical)
    compact["outcome_analysis"] = profile.get("outcome_analysis", {})
    
    # 3. Split Candidates (Critical for training rows policy)
    compact["split_candidates"] = profile.get("split_candidates", [])
    
    # 4. Leakage Flags (Critical for leakage policy)
    compact["leakage_flags"] = profile.get("leakage_flags", [])
    
    # 5. Missingness (Top 30 only)
    compact["missingness_top30"] = profile.get("missingness_top30", {})

    # 6. Sample Rows (Reduced)
    # If sample_rows exists, keep just a few columns or rows?
    # Usually sample_rows is separate in context, but if it's in profile, we might keep head.
    # Profile structure usually doesn't have sample_rows (Steward puts it in context separately).
    
    # 7. Column DTypes - Simplify
    # Instead of full list, provide summary counts or just names if < max_cols
    dtypes = profile.get("dtypes", {})
    if len(dtypes) > max_cols:
        # Too many columns, summarize
        type_counts = {}
        for col, dtype in dtypes.items():
            t = str(dtype)
            type_counts[t] = type_counts.get(t, 0) + 1
        compact["dtypes_summary"] = type_counts
        compact["dtypes_note"] = f"Total {len(dtypes)} columns. Showing only summary."
    else:
        compact["dtypes"] = dtypes

    return compact
