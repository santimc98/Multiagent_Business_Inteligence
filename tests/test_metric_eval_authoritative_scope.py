from src.utils.metric_eval import resolve_metric_value


def test_resolve_metric_value_prefers_holdout_over_ambiguous_primary_metric_value():
    resolved = resolve_metric_value(
        {
            "primary_metric_name": "qwk",
            "primary_metric_value": 0.7717,
            "holdout": {"qwk": 0.7626},
            "train": {"qwk": 0.7717},
        },
        "qwk",
    )

    assert resolved["value"] == 0.7626
    assert resolved["matched_key"] == "holdout.qwk"


def test_resolve_metric_value_keeps_explicit_primary_when_no_validation_scope():
    resolved = resolve_metric_value(
        {
            "primary_metric_name": "qwk",
            "primary_metric_value": 0.7717,
            "train": {"qwk": 0.7717},
        },
        "qwk",
    )

    assert resolved["value"] == 0.7717
    assert resolved["matched_key"] == "primary_metric_value"
