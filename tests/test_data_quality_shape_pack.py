from src.utils.data_quality_shape import (
    build_data_quality_shape_pack,
    summarize_data_quality_shape_pack,
)


def test_data_quality_shape_pack_detects_zero_null_and_concentration() -> None:
    profile = {
        "rows": 100,
        "cols": 3,
        "columns": ["exposure", "segment", "emptyish"],
        "type_hints": {
            "exposure": "numeric",
            "segment": "categorical",
            "emptyish": "numeric",
        },
        "missing_frac": {
            "exposure": 0.12,
            "segment": 0.0,
            "emptyish": 0.45,
        },
        "cardinality": {
            "exposure": {"unique": 8, "top_values": [{"value": "0", "count": 65}]},
            "segment": {"unique": 2, "top_values": [{"value": "A", "count": 90}]},
            "emptyish": {"unique": 1, "top_values": [{"value": "0", "count": 55}]},
        },
        "numeric_summary": {
            "exposure": {
                "count": 88,
                "mean": 10.0,
                "std": 4.0,
                "min": 0.0,
                "p01": 0.0,
                "q25": 0.0,
                "median": 0.0,
                "q75": 12.0,
                "p99": 99.0,
                "max": 120.0,
                "zero_frac": 0.65,
                "neg_frac": 0.0,
                "pos_frac": 0.35,
            },
            "emptyish": {
                "count": 55,
                "min": 0.0,
                "q25": 0.0,
                "median": 0.0,
                "q75": 0.0,
                "max": 0.0,
                "zero_frac": 0.55,
            },
        },
        "sampling": {"was_sampled": False, "sample_size": 100, "total_rows_in_file": 100},
    }

    pack = build_data_quality_shape_pack(profile)

    assert pack["role"] == "advisory_context_only"
    assert "must not reject" in pack["deterministic_policy"]
    signals = pack["shape_signals"]
    assert signals["counts"]["zero_missing_coexistence"] == 2
    assert signals["counts"]["high_top_value_concentration"] == 1
    assert signals["counts"]["constant_or_quasi_constant"] == 1

    exposure = next(item for item in pack["column_facts"] if item["name"] == "exposure")
    assert exposure["numeric"]["p99"] == 99.0
    assert "zero_missing_coexistence" in exposure["advisory_warnings"]
    assert "high_zero_fraction" in exposure["advisory_warnings"]


def test_data_quality_shape_summary_is_compact_and_advisory() -> None:
    pack = build_data_quality_shape_pack(
        {
            "rows": 10,
            "columns": ["flag"],
            "type_hints": {"flag": "numeric"},
            "missing_frac": {"flag": 0.1},
            "cardinality": {"flag": {"unique": 2, "top_values": [{"value": "0", "count": 9}]}},
            "numeric_summary": {"flag": {"count": 9, "zero_frac": 0.9, "min": 0, "q25": 0, "q75": 0, "max": 1}},
        }
    )

    summary = summarize_data_quality_shape_pack(pack)

    assert "DATA_QUALITY_SHAPE_PACK_SUMMARY" in summary
    assert "advisory_context_only" in summary
    assert "zero_missing_coexistence" in summary
    assert "not deterministic pass/fail" in summary
