import pandas as pd

from src.utils.leakage_sanity_audit import run_unsupervised_numeric_relation_audit


def test_leakage_audit_detects_sum_relation():
    data = {
        "a": list(range(50)),
        "b": list(range(50, 100)),
    }
    data["c"] = [data["a"][i] + data["b"][i] for i in range(50)]
    df = pd.DataFrame(data)

    audit = run_unsupervised_numeric_relation_audit(df, min_rows=30, tol=1e-9)
    relations = audit.get("relations", [])
    assert any(rel.get("type") == "sum" and set(rel.get("columns", [])) == {"a", "b", "c"} for rel in relations)
