"""
Dataset size estimation and scaling recommendations.

Provides universal utilities for handling large datasets efficiently.
"""

import os
from typing import Optional, Dict, Any


def file_size_mb(path: str) -> float:
    """
    Get file size in megabytes.

    Args:
        path: Path to file

    Returns:
        File size in MB
    """
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def estimate_rows_fast(path: str, encoding: str = "utf-8", max_bytes: int = 2_000_000) -> Optional[int]:
    """
    Fast estimate of CSV rows by reading initial bytes.

    Args:
        path: Path to CSV file
        encoding: File encoding (default: utf-8)
        max_bytes: Max bytes to read for estimation (default: 2MB)

    Returns:
        Estimated row count, or None if file can't be read
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            # Read up to max_bytes
            chunk = f.read(max_bytes)
            actual_bytes_read = len(chunk)
            
            # Count lines in chunk
            line_count = chunk.count("\n")

            if line_count == 0:
                return 0

            # Estimate total rows based on bytes actually read
            total_bytes = os.path.getsize(path)
            if actual_bytes_read >= total_bytes:
                # Read entire file, return actual count
                return line_count
            else:
                # Only read partial file, estimate based on ratio
                estimated_rows = int((line_count * total_bytes) / actual_bytes_read)
                return estimated_rows
    except Exception:
        return None


def classify_dataset_scale(file_mb: float, est_rows: Optional[int] = None) -> Dict[str, Any]:
    """
    Classify dataset scale and provide recommendations.

    Universal heuristics based on size and row count.

    Args:
        file_mb: File size in MB
        est_rows: Estimated row count (optional)

    Returns:
        Dict with:
        - scale: "small" | "medium" | "large"
        - max_train_rows: recommended max training rows
        - chunk_size: recommended chunk size for processing
        - prefer_parquet: whether to use parquet cache
    """
    if est_rows is None:
        est_rows = 0

    # Classification thresholds
    is_large = file_mb >= 200 or est_rows >= 800_000
    is_medium = file_mb >= 50 or est_rows >= 200_000

    if is_large:
        scale = "large"
        max_train_rows = 100_000
        chunk_size = 50_000
        prefer_parquet = True
    elif is_medium:
        scale = "medium"
        max_train_rows = 200_000
        chunk_size = 100_000
        prefer_parquet = True
    else:
        scale = "small"
        max_train_rows = 500_000
        chunk_size = None  # No chunking needed
        prefer_parquet = False

    return {
        "scale": scale,
        "max_train_rows": max_train_rows,
        "chunk_size": chunk_size,
        "prefer_parquet": prefer_parquet,
        "file_mb": file_mb,
        "est_rows": est_rows,
    }


def get_dataset_scale_hints(work_dir: str, cleaned_csv_path: str = "data/cleaned_data.csv") -> Dict[str, Any]:
    """
    Get dataset scale hints for ML engineer prompt.

    Args:
        work_dir: Working directory
        cleaned_csv_path: Relative path to cleaned CSV

    Returns:
        Dict with scale information and recommendations
    """
    full_path = os.path.join(work_dir, cleaned_csv_path)

    if not os.path.exists(full_path):
        return {
            "scale": "unknown",
            "max_train_rows": None,
            "chunk_size": None,
            "prefer_parquet": False,
            "file_mb": 0.0,
            "est_rows": None,
        }

    size_mb = file_size_mb(full_path)
    est_rows = estimate_rows_fast(full_path)

    return classify_dataset_scale(size_mb, est_rows)
