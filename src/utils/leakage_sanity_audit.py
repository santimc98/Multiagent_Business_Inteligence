import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _select_numeric_columns(df: pd.DataFrame, max_numeric_cols: int, min_rows: int) -> List[str]:
    numeric_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        non_null = df[col].notna().sum()
        if non_null >= min_rows:
            numeric_cols.append(col)
    # Prioritize by non-null count (descending) to maximize usable pairs
    numeric_cols = sorted(numeric_cols, key=lambda c: df[c].notna().sum(), reverse=True)
    return numeric_cols[:max_numeric_cols]


def _identity_or_scale(a: pd.Series, b: pd.Series, tol: float) -> Optional[Dict[str, float]]:
    aligned = pd.concat([a, b], axis=1).dropna()
    if aligned.empty:
        return None
    diff = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs()
    if diff.max() <= tol:
        return {"type": "identity", "support": len(aligned)}
    # Scale detection (single factor)
    denom = aligned.iloc[:, 1].replace(0, np.nan)
    ratio = aligned.iloc[:, 0] / denom
    ratio = ratio.dropna()
    if ratio.empty:
        return None
    median_ratio = ratio.median()
    if median_ratio == 0:
        return None
    recon = aligned.iloc[:, 1] * median_ratio
    scale_err = (recon - aligned.iloc[:, 0]).abs().max()
    if scale_err <= tol:
        return {"type": "scale", "support": len(recon), "scale": float(median_ratio)}
    return None


def _sum_diff_match(a: pd.Series, b: pd.Series, target: pd.Series, tol: float) -> Optional[Dict[str, float]]:
    aligned = pd.concat([a, b, target], axis=1).dropna()
    if aligned.empty:
        return None
    sum_err = (aligned.iloc[:, 0] + aligned.iloc[:, 1] - aligned.iloc[:, 2]).abs()
    if sum_err.max() <= tol:
        return {"type": "sum", "support": len(sum_err)}
    diff_err = (aligned.iloc[:, 0] - aligned.iloc[:, 1] - aligned.iloc[:, 2]).abs()
    if diff_err.max() <= tol:
        return {"type": "diff", "support": len(diff_err)}
    return None


def run_unsupervised_numeric_relation_audit(
    df: pd.DataFrame,
    min_rows: int = 30,
    max_numeric_cols: int = 30,
    max_pairs: int = 1200,
    tol: float = 1e-9,
    frac: float = 0.995,
) -> Dict[str, object]:
    """
    Detects near-deterministic numeric relations (identity, scale, sum, diff) across the dataset.
    Returns a dict with discovered relations; does not raise.
    """
    findings: List[Dict[str, object]] = []
    numeric_cols = _select_numeric_columns(df, max_numeric_cols, min_rows)
    n = len(numeric_cols)
    total_rows = len(df)

    pair_budget = max_pairs
    # Identity / scale (pairwise)
    for i in range(n):
        for j in range(i + 1, n):
            if pair_budget <= 0:
                break
            c1, c2 = numeric_cols[i], numeric_cols[j]
            res = _identity_or_scale(df[c1], df[c2], tol)
            pair_budget -= 1
            if res:
                support_frac = res["support"] / max(total_rows, 1)
                if support_frac >= frac:
                    findings.append(
                        {
                            "type": res["type"],
                            "columns": [c1, c2],
                            "support": res["support"],
                            "support_frac": support_frac,
                            "scale": res.get("scale"),
                        }
                    )
        if pair_budget <= 0:
            break

    # Sum/Diff (triples) within remaining budget
    triple_budget = max_pairs
    for target_col in numeric_cols:
        if triple_budget <= 0:
            break
        for i in range(n):
            if numeric_cols[i] == target_col:
                continue
            if triple_budget <= 0:
                break
            for j in range(i + 1, n):
                if numeric_cols[j] == target_col:
                    continue
                c1, c2 = numeric_cols[i], numeric_cols[j]
                res = _sum_diff_match(df[c1], df[c2], df[target_col], tol)
                triple_budget -= 1
                if res:
                    support_frac = res["support"] / max(total_rows, 1)
                    if support_frac >= frac:
                        findings.append(
                            {
                                "type": res["type"],
                                "columns": [c1, c2, target_col],
                                "support": res["support"],
                                "support_frac": support_frac,
                            }
                        )
            if triple_budget <= 0:
                break

    summary = {"relations": findings, "scanned_columns": numeric_cols, "rows": total_rows}
    return summary


def assert_no_deterministic_target_leakage(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    min_rows: int = 30,
    max_pairs: int = 1200,
    tol: float = 1e-9,
    frac: float = 0.995,
) -> None:
    """
    Raises ValueError if the target is (near) deterministically derived from features
    via identity, scale, sum, or diff relations.
    """
    if target_col not in df.columns:
        return
    target_series = pd.to_numeric(df[target_col], errors="coerce")
    target_valid = target_series.dropna()
    if len(target_valid) < min_rows:
        return

    features = []
    for col in feature_cols:
        if col == target_col or col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() >= min_rows:
            features.append(col)

    if not features:
        return

    total_rows = len(target_series)
    # Identity / scale
    pair_budget = max_pairs
    for feat in features:
        aligned = pd.concat([target_series, pd.to_numeric(df[feat], errors="coerce")], axis=1).dropna()
        if len(aligned) < min_rows:
            continue
        res = _identity_or_scale(aligned.iloc[:, 0], aligned.iloc[:, 1], tol)
        pair_budget -= 1
        if res:
            support_frac = res["support"] / max(total_rows, 1)
            if support_frac >= frac:
                raise ValueError(
                    f"DETERMINISTIC_TARGET_RELATION: target ~ {feat} ({res['type']}), support_frac={support_frac:.3f}"
                )
        if pair_budget <= 0:
            return

    # Sum/Diff (triples)
    triple_budget = max_pairs
    m = len(features)
    for i in range(m):
        if triple_budget <= 0:
            break
        for j in range(i + 1, m):
            if triple_budget <= 0:
                break
            f1, f2 = features[i], features[j]
            aligned = pd.concat(
                [
                    target_series,
                    pd.to_numeric(df[f1], errors="coerce"),
                    pd.to_numeric(df[f2], errors="coerce"),
                ],
                axis=1,
            ).dropna()
            if len(aligned) < min_rows:
                triple_budget -= 1
                continue
            res = _sum_diff_match(aligned.iloc[:, 1], aligned.iloc[:, 2], aligned.iloc[:, 0], tol)
            triple_budget -= 1
            if res:
                support_frac = res["support"] / max(total_rows, 1)
                if support_frac >= frac:
                    relation = f"{f1} {res['type']} {f2}"
                    raise ValueError(
                        f"DETERMINISTIC_TARGET_RELATION: target ~ {relation}, support_frac={support_frac:.3f}"
                    )
