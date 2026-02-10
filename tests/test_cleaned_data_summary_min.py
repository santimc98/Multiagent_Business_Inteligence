import pandas as pd

from src.graph.graph import _build_cleaned_data_summary_min


def test_cleaned_data_summary_min_flags_missing_required_and_role_dtype_warnings():
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "vendor_id": [1, 2, 1, 2],
            "pickup_datetime": [
                "2026-01-01 10:00:00",
                "2026-01-01 10:05:00",
                "2026-01-01 10:10:00",
                "2026-01-01 10:15:00",
            ],
            "trip_duration": [300.0, 450.0, None, 700.0],
            "__split": ["train", "train", "test", "test"],
        }
    )
    contract = {
        "canonical_columns": ["id", "vendor_id", "pickup_datetime", "trip_duration", "__split"],
        "column_roles": {
            "id": ["id"],
            "categorical_features": ["vendor_id"],
            "temporal_features": ["pickup_datetime"],
            "target": ["trip_duration"],
            "split_indicator": ["__split"],
        },
        "objective_analysis": {"problem_type": "regression"},
    }
    required_columns = ["id", "vendor_id", "pickup_datetime", "trip_duration", "__split", "store_and_fwd_flag"]

    summary = _build_cleaned_data_summary_min(
        df_clean=df,
        contract=contract,
        required_columns=required_columns,
        data_path="data/cleaned_trips.csv",
    )

    assert "store_and_fwd_flag" in summary.get("missing_required_columns", [])
    assert summary.get("split_column") == "__split"

    by_name = {entry["column_name"]: entry for entry in summary.get("column_summaries", [])}
    vendor = by_name["vendor_id"]
    assert vendor["mismatch_with_contract"]["role_dtype_warning"] == "categorical_role_with_numeric_dtype"
    assert vendor["in_train"]["rows"] == 2
    assert vendor["in_test"]["rows"] == 2

