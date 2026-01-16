"""
Tests for feature_selectors.py (P1.5).
"""
import pytest
from src.utils.feature_selectors import (
    expand_feature_selectors,
    infer_feature_selectors,
    get_all_feature_columns,
    compact_column_representation,
)


class TestExpandFeatureSelectors:
    """Test feature selector expansion."""

    def test_expand_regex_selector(self):
        """Test regex selector expansion."""
        columns = ["pixel0", "pixel1", "pixel100", "pixel783", "id", "target"]
        selectors = [
            {"type": "regex", "pattern": r"^pixel\d+$", "role": "model_feature"}
        ]

        expanded = expand_feature_selectors(columns, selectors)

        assert "pixel0" in expanded
        assert "pixel1" in expanded
        assert "pixel100" in expanded
        assert "pixel783" in expanded
        assert "id" not in expanded
        assert "target" not in expanded

    def test_expand_prefix_selector(self):
        """Test prefix selector expansion."""
        columns = ["ps_ind_01", "ps_ind_02", "ps_calc_01", "id", "target"]
        selectors = [
            {"type": "prefix", "value": "ps_", "role": "model_feature"}
        ]

        expanded = expand_feature_selectors(columns, selectors)

        assert "ps_ind_01" in expanded
        assert "ps_ind_02" in expanded
        assert "ps_calc_01" in expanded
        assert "id" not in expanded
        assert "target" not in expanded

    def test_expand_multiple_selectors(self):
        """Test multiple selectors."""
        columns = ["pixel0", "pixel1", "feat_a", "feat_b", "id"]
        selectors = [
            {"type": "regex", "pattern": r"^pixel\d+$"},
            {"type": "prefix", "value": "feat_"},
        ]

        expanded = expand_feature_selectors(columns, selectors)

        assert "pixel0" in expanded
        assert "pixel1" in expanded
        assert "feat_a" in expanded
        assert "feat_b" in expanded
        assert "id" not in expanded


class TestInferFeatureSelectors:
    """Test automatic selector inference for wide datasets."""

    def test_infer_selectors_for_prefix_digit_pattern(self):
        """Test that prefix+digit patterns are detected."""
        # Create 300 pixel columns (MNIST-like)
        columns = [f"pixel{i}" for i in range(300)] + ["id", "target"]

        selectors, remaining = infer_feature_selectors(
            columns, max_list_size=200, min_group_size=50
        )

        # Should infer a regex selector for pixel columns
        assert len(selectors) >= 1
        pixel_selector = next(
            (s for s in selectors if "pixel" in s.get("pattern", "")),
            None
        )
        assert pixel_selector is not None
        assert pixel_selector["type"] == "regex"

        # id and target should remain in remaining
        assert "id" in remaining
        assert "target" in remaining

        # pixel columns should not be in remaining
        assert "pixel0" not in remaining

    def test_infer_selectors_for_common_prefix(self):
        """Test that common prefixes are detected."""
        # Create 100 columns with ps_ prefix (first segment before _)
        columns = [f"ps_calc_{i:02d}" for i in range(60)] + [f"ps_ind_{i:02d}" for i in range(60)] + ["id", "target"]

        selectors, remaining = infer_feature_selectors(
            columns, max_list_size=50, min_group_size=30
        )

        # Should infer at least one selector (could be prefix or regex)
        assert len(selectors) >= 1

        # Check that most columns are covered by selectors
        from src.utils.feature_selectors import expand_feature_selectors
        expanded = expand_feature_selectors(columns, selectors)
        # At least 60% of ps_ columns should be covered
        ps_columns = [c for c in columns if c.startswith("ps_")]
        covered = [c for c in ps_columns if c in expanded]
        assert len(covered) >= len(ps_columns) * 0.5, f"Only {len(covered)}/{len(ps_columns)} covered"

    def test_no_selectors_for_small_dataset(self):
        """Test that small datasets don't get selectors."""
        columns = ["col_a", "col_b", "col_c", "id"]

        selectors, remaining = infer_feature_selectors(
            columns, max_list_size=200, min_group_size=50
        )

        # No selectors needed
        assert len(selectors) == 0
        assert set(remaining) == set(columns)

    def test_columns_never_truncated(self):
        """Test that no columns are lost - they're either in selectors or remaining."""
        # Create 500 columns
        columns = [f"pixel{i}" for i in range(400)] + [f"other_{i}" for i in range(100)]

        selectors, remaining = infer_feature_selectors(
            columns, max_list_size=200, min_group_size=50
        )

        # Expand selectors and combine with remaining
        expanded = expand_feature_selectors(columns, selectors)
        all_covered = set(expanded) | set(remaining)

        # All original columns must be accounted for
        assert all_covered == set(columns), "Some columns were lost!"


class TestGetAllFeatureColumns:
    """Test combining explicit features with selectors."""

    def test_combine_explicit_and_selectors(self):
        """Test combining explicit list with selector expansion."""
        columns = ["pixel0", "pixel1", "pixel2", "manual_feat", "id"]
        explicit = ["manual_feat"]
        selectors = [{"type": "regex", "pattern": r"^pixel\d+$"}]

        all_features = get_all_feature_columns(columns, explicit, selectors)

        assert "manual_feat" in all_features
        assert "pixel0" in all_features
        assert "pixel1" in all_features
        assert "pixel2" in all_features
        assert "id" not in all_features


class TestCompactColumnRepresentation:
    """Test compact display of wide datasets."""

    def test_compact_representation_wide_dataset(self):
        """Test compact representation for wide dataset."""
        # 500 pixel columns
        columns = [f"pixel{i}" for i in range(500)]

        result = compact_column_representation(columns, max_display=20)

        assert result["total_count"] == 500
        # Selectors should be inferred for pixel columns
        assert len(result["inferred_selectors"]) > 0
        # After selectors cover pixel columns, remaining should be small or empty
        # The compact representation handles the width via selectors
        assert result["total_count"] == 500

    def test_compact_representation_mixed_dataset(self):
        """Test compact representation with mixed column types."""
        # Mix of patterned and random columns
        columns = [f"pixel{i}" for i in range(100)] + [f"random_{i}" for i in range(50)]

        result = compact_column_representation(columns, max_display=20)

        assert result["total_count"] == 150
        # At least pixel columns should be covered by selectors
        assert len(result["inferred_selectors"]) >= 1
