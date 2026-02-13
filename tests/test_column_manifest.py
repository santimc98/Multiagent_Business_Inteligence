from src.utils.column_sets import (
    build_column_sets,
    build_column_manifest,
    summarize_column_manifest,
)


def test_build_column_manifest_wide_from_column_sets() -> None:
    columns = ["label", "__split"] + [f"pixel{i}" for i in range(784)]
    roles = {"label": "target_candidate", "__split": "split_candidate"}
    sets_spec = build_column_sets(columns, roles=roles)
    manifest = build_column_manifest(columns, column_sets=sets_spec, roles=roles)

    assert manifest.get("schema_mode") == "wide"
    assert manifest.get("total_columns") == len(columns)
    anchors = manifest.get("anchors") or []
    assert "label" in anchors
    assert "__split" in anchors

    families = manifest.get("families") or []
    assert families, "Expected at least one feature family in wide schema mode"
    first_selector = families[0].get("selector") if isinstance(families[0], dict) else {}
    assert isinstance(first_selector, dict)
    assert first_selector.get("type") in {"prefix_numeric_range", "regex", "all_columns_except"}


def test_summarize_column_manifest_non_empty() -> None:
    columns = ["target"] + [f"feat{i}" for i in range(220)]
    sets_spec = build_column_sets(columns, roles={"target": "target_candidate"})
    manifest = build_column_manifest(columns, column_sets=sets_spec, roles={"target": "target_candidate"})
    summary = summarize_column_manifest(manifest)
    assert "COLUMN_MANIFEST_SUMMARY" in summary
    assert "schema_mode" in summary
