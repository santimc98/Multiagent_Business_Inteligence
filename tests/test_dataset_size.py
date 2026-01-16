"""
Tests for dataset_size module.

Tests file size estimation, row counting, and scale classification.
"""

import pytest
import tempfile
from pathlib import Path

from src.utils.dataset_size import (
    file_size_mb,
    estimate_rows_fast,
    classify_dataset_scale,
    get_dataset_scale_hints,
)


def test_file_size_mb(tmp_path: Path):
    """Test file size calculation in megabytes."""
    # Create test file (1 MB = 1024 * 1024 bytes)
    test_file = tmp_path / "test.csv"
    test_file.write_bytes(b"x" * (1024 * 1024))

    size_mb = file_size_mb(str(test_file))

    assert size_mb >= 0.99 and size_mb <= 1.01


def test_estimate_rows_fast_small_csv():
    """Test row estimation on small CSV."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "test.csv"
        # Create small CSV (100 rows)
        content = "col1,col2\n" + "\n".join([f"val{i},{i*2}" for i in range(100)])
        test_file.write_text(content, encoding="utf-8", newline="\n")

        est_rows = estimate_rows_fast(str(test_file))

        assert est_rows >= 90 and est_rows <= 110


def test_classify_dataset_scale_small():
    """Test scale classification for small dataset."""
    scale_info = classify_dataset_scale(file_mb=10, est_rows=50_000)

    assert scale_info["scale"] == "small"
    assert scale_info["max_train_rows"] == 500_000
    assert scale_info["chunk_size"] is None
    assert scale_info["prefer_parquet"] is False


def test_classify_dataset_scale_medium():
    """Test scale classification for medium dataset."""
    scale_info = classify_dataset_scale(file_mb=100, est_rows=500_000)

    assert scale_info["scale"] == "medium"
    assert scale_info["max_train_rows"] == 200_000
    assert scale_info["chunk_size"] == 100_000
    assert scale_info["prefer_parquet"] is True


def test_classify_dataset_scale_large():
    """Test scale classification for large dataset."""
    scale_info = classify_dataset_scale(file_mb=250, est_rows=1_000_000)

    assert scale_info["scale"] == "large"
    assert scale_info["max_train_rows"] == 100_000
    assert scale_info["chunk_size"] == 50_000
    assert scale_info["prefer_parquet"] is True


def test_get_dataset_scale_hints():
    """Test getting dataset scale hints from a work directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir) / "work_dir"
        work_dir.mkdir()

        data_dir = work_dir / "data"
        data_dir.mkdir()

        # Create small CSV
        csv_file = data_dir / "cleaned_data.csv"
        content = "id,value\n" + "\n".join([f"{i},{i*10}" for i in range(10)])
        csv_file.write_text(content, encoding="utf-8", newline="\n")

        hints = get_dataset_scale_hints(str(work_dir), "data/cleaned_data.csv")

        assert hints["scale"] == "small"
        assert hints["file_mb"] >= 0 and hints["file_mb"] < 0.01
        assert hints["est_rows"] >= 5 and hints["est_rows"] <= 20
        assert hints["max_train_rows"] == 500_000
