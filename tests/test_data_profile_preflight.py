"""
Tests for data_profile_preflight module.

Tests that ensure_data_profile_artifact:
  1) Converts dataset_profile.json when available
  2) Creates minimal fallback when dataset_profile is missing
  3) Is idempotent (doesn't rewrite if already present)
"""
import json
import os
import tempfile
import pytest
from src.utils.data_profile_preflight import ensure_data_profile_artifact


class TestEnsureDataProfileArtifact:
    """Test ensure_data_profile_artifact function."""

    def test_convert_from_dataset_profile(self, tmp_path):
        """When dataset_profile.json exists, it should be converted."""
        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create dataset_profile.json
        dataset_profile = {
            "rows": 1000,
            "cols": 10,
            "columns": ["Id", "Feature1", "Feature2", "Target"],
            "missing_frac": {"Id": 0.0, "Feature1": 0.1, "Feature2": 0.05, "Target": 0.0},
            "cardinality": {
                "Id": {"unique": 1000, "top_values": []},
                "Feature1": {"unique": 50, "top_values": []},
                "Feature2": {"unique": 100, "top_values": []},
                "Target": {"unique": 2, "top_values": [{"value": "0", "count": 600}, {"value": "1", "count": 400}]},
            },
            "type_hints": {"Id": "numeric", "Feature1": "numeric", "Feature2": "categorical", "Target": "categorical"},
        }
        (data_dir / "dataset_profile.json").write_text(json.dumps(dataset_profile))

        # Create contract with outcome column
        contract = {
            "outcome_columns": ["Target"],
            "column_roles": {"outcome": ["Target"]},
        }

        state = {}

        # Call the function
        result, source = ensure_data_profile_artifact(
            state=state,
            contract=contract,
            analysis_type="classification",
            work_dir_abs=str(tmp_path),
            run_id=None,
        )

        # Verify
        assert source == "converted_from_dataset_profile"
        assert result is not None
        assert result.get("basic_stats", {}).get("n_rows") == 1000
        assert result.get("basic_stats", {}).get("n_cols") == 10
        assert "Target" in result.get("outcome_analysis", {})

        # Verify state was updated
        assert state.get("data_profile") == result
        assert state.get("data_profile_path") == "data/data_profile.json"
        assert state.get("data_profile_source") == "converted_from_dataset_profile"

        # Verify file was written
        output_path = data_dir / "data_profile.json"
        assert output_path.exists()

    def test_minimal_fallback_no_dataset_profile(self, tmp_path):
        """When dataset_profile.json is missing, create minimal fallback."""
        # Create data directory (but no dataset_profile.json)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Contract with some info
        contract = {
            "canonical_columns": ["Id", "Feature1", "Target"],
            "outcome_columns": ["Target"],
        }

        state = {}

        # Call the function
        result, source = ensure_data_profile_artifact(
            state=state,
            contract=contract,
            analysis_type="classification",
            work_dir_abs=str(tmp_path),
            run_id=None,
        )

        # Verify
        assert source == "minimal_fallback"
        assert result is not None
        assert result.get("source") == "minimal_fallback_no_dataset_profile"
        assert result.get("basic_stats", {}).get("n_rows") == 0  # Unknown
        assert "Target" in result.get("outcome_analysis", {})

        # Verify state was updated
        assert state.get("data_profile") is not None
        assert state.get("data_profile_source") == "minimal_fallback"

        # Verify file was written
        output_path = data_dir / "data_profile.json"
        assert output_path.exists()

    def test_idempotent_already_present(self, tmp_path):
        """If data_profile already exists in state and file, return early."""
        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create existing data_profile.json
        existing_profile = {
            "basic_stats": {"n_rows": 500, "n_cols": 5, "columns": ["A", "B"]},
            "outcome_analysis": {},
            "source": "pre_existing",
        }
        (data_dir / "data_profile.json").write_text(json.dumps(existing_profile))

        # State already has the profile
        state = {
            "data_profile": existing_profile,
            "data_profile_path": "data/data_profile.json",
        }

        contract = {}

        # Call the function
        result, source = ensure_data_profile_artifact(
            state=state,
            contract=contract,
            analysis_type=None,
            work_dir_abs=str(tmp_path),
            run_id=None,
        )

        # Verify it returned early without rewriting
        assert source == "already_present"
        assert result == existing_profile

    def test_column_inventory_fallback(self, tmp_path):
        """When no dataset_profile, use column_inventory from state."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        contract = {}

        # State has column_inventory (dict format)
        state = {
            "column_inventory": {
                "n_columns": 3,
                "columns": ["Col1", "Col2", "Col3"],
            }
        }

        result, source = ensure_data_profile_artifact(
            state=state,
            contract=contract,
            analysis_type=None,
            work_dir_abs=str(tmp_path),
            run_id=None,
        )

        assert source == "minimal_fallback"
        assert result.get("basic_stats", {}).get("columns") == ["Col1", "Col2", "Col3"]
        assert result.get("basic_stats", {}).get("n_cols") == 3

    def test_column_inventory_list_format(self, tmp_path):
        """When column_inventory is a list, use it directly."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        contract = {}

        # State has column_inventory (list format)
        state = {
            "column_inventory": ["A", "B", "C", "D"],
        }

        result, source = ensure_data_profile_artifact(
            state=state,
            contract=contract,
            analysis_type=None,
            work_dir_abs=str(tmp_path),
            run_id=None,
        )

        assert source == "minimal_fallback"
        assert result.get("basic_stats", {}).get("columns") == ["A", "B", "C", "D"]
        assert result.get("basic_stats", {}).get("n_cols") == 4

    def test_no_crash_on_empty_inputs(self, tmp_path):
        """Empty state/contract should not crash, just create minimal profile."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        state = {}
        contract = {}

        result, source = ensure_data_profile_artifact(
            state=state,
            contract=contract,
            analysis_type=None,
            work_dir_abs=str(tmp_path),
            run_id=None,
        )

        # Should not crash
        assert result is not None
        assert source == "minimal_fallback"
        assert result.get("basic_stats", {}).get("columns") == []

    def test_invalid_dataset_profile_falls_back(self, tmp_path):
        """If dataset_profile.json exists but is invalid, fall back to minimal."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create invalid dataset_profile (missing required fields)
        invalid_profile = {"some_key": "some_value"}
        (data_dir / "dataset_profile.json").write_text(json.dumps(invalid_profile))

        state = {}
        contract = {"canonical_columns": ["X", "Y"]}

        result, source = ensure_data_profile_artifact(
            state=state,
            contract=contract,
            analysis_type=None,
            work_dir_abs=str(tmp_path),
            run_id=None,
        )

        # Should fall back to minimal
        assert source == "minimal_fallback"
        assert result.get("basic_stats", {}).get("columns") == ["X", "Y"]


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_outcome_analysis_populated(self, tmp_path):
        """Outcome analysis should be populated from contract."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create valid dataset_profile
        dataset_profile = {
            "rows": 100,
            "cols": 3,
            "columns": ["Id", "Survived"],
            "missing_frac": {"Id": 0.0, "Survived": 0.2},
            "cardinality": {
                "Id": {"unique": 100, "top_values": []},
                "Survived": {"unique": 2, "top_values": [{"value": "0", "count": 60}, {"value": "1", "count": 40}]},
            },
            "type_hints": {"Id": "numeric", "Survived": "categorical"},
        }
        (data_dir / "dataset_profile.json").write_text(json.dumps(dataset_profile))

        contract = {
            "outcome_columns": ["Survived"],
        }

        state = {}

        result, source = ensure_data_profile_artifact(
            state=state,
            contract=contract,
            analysis_type="classification",
            work_dir_abs=str(tmp_path),
            run_id=None,
        )

        assert source == "converted_from_dataset_profile"
        outcome = result.get("outcome_analysis", {}).get("Survived", {})
        assert outcome.get("present") is True
        assert outcome.get("inferred_type") == "classification"
        assert outcome.get("n_unique") == 2

    def test_file_persisted_for_run_bundle(self, tmp_path):
        """The data_profile.json file should be persisted for run bundle capture."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        state = {"column_inventory": ["A", "B"]}
        contract = {}

        result, source = ensure_data_profile_artifact(
            state=state,
            contract=contract,
            analysis_type=None,
            work_dir_abs=str(tmp_path),
            run_id=None,
        )

        # Verify file exists
        output_path = data_dir / "data_profile.json"
        assert output_path.exists()

        # Verify it's valid JSON
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded.get("basic_stats", {}).get("columns") == ["A", "B"]
