"""
Tests for sandbox_paths module.

Tests universal path standardization and alias handling.
"""

import pytest
import tempfile
from pathlib import Path

from src.utils.sandbox_paths import (
    CANONICAL_RAW_REL,
    CANONICAL_CLEANED_REL,
    CANONICAL_MANIFEST_REL,
    COMMON_RAW_ALIASES,
    COMMON_CLEANED_ALIASES,
    patch_placeholders,
    build_symlink_or_copy_commands,
    canonical_abs,
)


def test_patch_placeholders_replaces_data_path():
    """Test that $data_path is replaced correctly."""
    code = 'pd.read_csv("$data_path")'
    patched = patch_placeholders(code, data_rel="data/my_file.csv")

    assert "$data_path" not in patched
    assert "data/my_file.csv" in patched


def test_patch_placeholders_replaces_manifest_path():
    """Test that $manifest_path is replaced correctly."""
    code = 'manifest = json.load("$manifest_path")'
    patched = patch_placeholders(code, manifest_rel="data/my_manifest.json")

    assert "$manifest_path" not in patched
    assert "data/my_manifest.json" in patched


def test_build_commands_root_aliases_contains_ln_or_cp():
    """Test that root alias commands contain ln or cp fallback."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_root = str(Path(tmp_dir) / "run_root")

        commands = build_symlink_or_copy_commands(
            run_root,
            canonical_rel=CANONICAL_CLEANED_REL,
            aliases=["data.csv", "cleaned.csv"]
        )

        assert len(commands) > 0

        full_cmd = " && ".join(commands)
        assert "ln -sf data/cleaned_data.csv" in full_cmd
        assert "cp -f data/cleaned_data.csv" in full_cmd
        assert "mkdir -p" in commands[0]


def test_build_commands_nested_alias_uses_cd_data():
    """Test that nested aliases in data/ use cd data with correct symlinks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_root = str(Path(tmp_dir) / "run_root")

        commands = build_symlink_or_copy_commands(
            run_root,
            canonical_rel=CANONICAL_RAW_REL,
            aliases=["data/raw_data.csv", "data/input.csv"]
        )

        assert len(commands) > 0

        # Verify structure: mkdir, cd to run_root, ln for root aliases, cd to data, ln for data/ aliases
        full_cmd = " && ".join(commands)
        assert "cd" in full_cmd
        assert "ln -sf raw.csv raw_data.csv" in full_cmd
        assert "ln -sf raw.csv input.csv" in full_cmd
        # Verify we cd into data directory for nested aliases
        assert f"{run_root}/data" in full_cmd


def test_canonical_abs_avoids_double_slashes():
    """Test that canonical_abs avoids double slashes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_root = f"{tmp_dir}/run_root"  # Has trailing slash

        abs_path = canonical_abs(run_root, "data/cleaned.csv")

        assert abs_path == f"{tmp_dir}/run_root/data/cleaned.csv"
        assert "run_root//data" not in abs_path
