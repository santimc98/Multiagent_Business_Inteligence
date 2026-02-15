from __future__ import annotations

import re
from typing import Any, Dict, List


_DOMAIN_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "domain": "finance_insurance",
        "keywords": [
            "insurance",
            "credit",
            "loan",
            "policy",
            "pricing",
            "fraud",
            "underwriting",
            "premium",
            "bank",
            "risk score",
        ],
        "best_practices": [
            "Prioritize calibration and rank stability for risk decisioning.",
            "Demand transparent decision thresholds and confidence handling.",
            "Track fairness slices for protected or proxy groups where possible.",
        ],
        "risk_watchlist": [
            "Data leakage from post-outcome variables.",
            "Unstable thresholds across segments or time periods.",
            "Overfitting on sparse high-cardinality identifiers.",
        ],
    },
    {
        "domain": "healthcare_life_sciences",
        "keywords": [
            "patient",
            "clinical",
            "hospital",
            "diagnosis",
            "medical",
            "treatment",
            "survival",
            "adverse event",
        ],
        "best_practices": [
            "Favor interpretable features and robust uncertainty reporting.",
            "Validate across cohorts and temporal slices to avoid drift harm.",
            "Use conservative thresholds when false negatives are high impact.",
        ],
        "risk_watchlist": [
            "Dataset shift between institutions or time windows.",
            "Proxy bias from demographic or access variables.",
            "Overclaiming causal interpretation from observational data.",
        ],
    },
    {
        "domain": "retail_marketing",
        "keywords": [
            "customer",
            "churn",
            "campaign",
            "basket",
            "purchase",
            "conversion",
            "promotion",
            "lifetime value",
        ],
        "best_practices": [
            "Optimize for business utility (uplift, precision@k, revenue impact).",
            "Separate training and scoring populations to avoid target leakage.",
            "Provide actionable segment-level drivers, not only global metrics.",
        ],
        "risk_watchlist": [
            "Label leakage from campaign-response artifacts.",
            "Temporal leakage between pre/post intervention periods.",
            "Metric mismatch between leaderboard KPI and business objective.",
        ],
    },
    {
        "domain": "mobility_logistics",
        "keywords": [
            "trip",
            "taxi",
            "eta",
            "route",
            "delivery",
            "pickup",
            "dropoff",
            "distance",
            "traffic",
        ],
        "best_practices": [
            "Account for temporal and geospatial non-stationarity.",
            "Use robust error metrics and inspect tail-risk predictions.",
            "Validate latency and scalability constraints for real-time scoring.",
        ],
        "risk_watchlist": [
            "Spatiotemporal leakage from future-derived features.",
            "Long-tail errors hidden by average metrics.",
            "Runtime bottlenecks from expensive feature engineering.",
        ],
    },
    {
        "domain": "manufacturing_operations",
        "keywords": [
            "sensor",
            "machine",
            "failure",
            "maintenance",
            "downtime",
            "quality control",
            "defect",
        ],
        "best_practices": [
            "Prioritize robust anomaly/failure recall with practical false alarm control.",
            "Check concept drift and recalibration cadence for production stability.",
            "Use conservative thresholds for safety-critical failure prediction.",
        ],
        "risk_watchlist": [
            "Data imbalance masking rare but critical failures.",
            "Unreliable labels due to delayed or noisy failure annotation.",
            "Hidden leakage from maintenance logs generated after incident.",
        ],
    },
]


def _normalize_text(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", str(text or "").lower()))


def infer_domain_guidance(
    *,
    data_summary: str,
    business_objective: str,
    dataset_memory_context: str = "",
    max_domains: int = 2,
) -> Dict[str, Any]:
    source = " ".join(
        [
            _normalize_text(data_summary),
            _normalize_text(business_objective),
            _normalize_text(dataset_memory_context),
        ]
    )
    if not source.strip():
        return {
            "inferred_domains": [],
            "best_practices": [],
            "risk_watchlist": [],
            "confidence": "low",
        }

    scored: List[Dict[str, Any]] = []
    for entry in _DOMAIN_KNOWLEDGE_BASE:
        keywords = [str(k).lower() for k in (entry.get("keywords") or []) if k]
        hits = [kw for kw in keywords if kw in source]
        if not hits:
            continue
        scored.append(
            {
                "domain": entry.get("domain"),
                "hit_count": len(hits),
                "hits": hits[:12],
                "best_practices": entry.get("best_practices") or [],
                "risk_watchlist": entry.get("risk_watchlist") or [],
            }
        )
    scored.sort(key=lambda item: int(item.get("hit_count") or 0), reverse=True)
    selected = scored[: max(1, int(max_domains))]

    practices: List[str] = []
    risks: List[str] = []
    for item in selected:
        for p in item.get("best_practices") or []:
            if p not in practices:
                practices.append(str(p))
        for r in item.get("risk_watchlist") or []:
            if r not in risks:
                risks.append(str(r))

    confidence = "high" if selected and int(selected[0].get("hit_count") or 0) >= 3 else "medium" if selected else "low"
    return {
        "inferred_domains": selected,
        "best_practices": practices[:10],
        "risk_watchlist": risks[:12],
        "confidence": confidence,
    }
