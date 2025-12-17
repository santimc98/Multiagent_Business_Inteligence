import pandas as pd
import pytest

from src.utils.leakage_sanity_audit import assert_no_deterministic_target_leakage


def test_guardrail_blocks_target_tautology():
    df = pd.DataFrame(
        {
            "f1": list(range(60)),
            "f2": [v * 2 for v in range(60)],
        }
    )
    df["target"] = df["f1"] + df["f2"]

    with pytest.raises(ValueError) as excinfo:
        assert_no_deterministic_target_leakage(df, "target", ["f1", "f2"], tol=1e-9, frac=0.99)

    assert "DETERMINISTIC_TARGET_RELATION" in str(excinfo.value)
