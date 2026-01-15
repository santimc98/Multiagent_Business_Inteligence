#!/usr/bin/env python3
"""
Combine Kaggle-style train/test CSV files into a single CSV for pipelines that accept only one file.

Output format:
- Adds a column __split with values: "train" or "test"
- Ensures both sides have identical columns
- Ensures target exists as a column on both sides (NaN for test)
- Writes: data/combined_<tag>.csv

Usage examples:
  python combine_kaggle_train_test.py --train data/train.csv --test data/test.csv --out data/combined_titanic.csv --id PassengerId --target Survived
  python combine_kaggle_train_test.py --train data/train.csv --test data/test.csv --out data/combined_house_prices.csv --id Id --target SalePrice
  python combine_kaggle_train_test.py --train data/train.csv --test data/test.csv --out data/combined_auto.csv --id Id
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd


SPLIT_COL = "__split"
TRAIN_TAG = "train"
TEST_TAG = "test"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # Use engine="python" to be more tolerant with weird separators; pandas will still infer delimiter if not set.
    # If you know the delimiter, pass it explicitly (not done here to keep universal).
    return pd.read_csv(path)


def _infer_target_column(train: pd.DataFrame, test: pd.DataFrame, id_col: Optional[str]) -> Optional[str]:
    """
    Infer target as a column present in train but missing in test (excluding obvious split/id).
    Returns:
      - the inferred target column name if exactly one candidate exists
      - None otherwise
    """
    train_cols = set(train.columns)
    test_cols = set(test.columns)

    candidates = list(train_cols - test_cols)
    # Remove split/id if they somehow exist
    if SPLIT_COL in candidates:
        candidates.remove(SPLIT_COL)
    if id_col and id_col in candidates:
        candidates.remove(id_col)

    # Common Kaggle pattern: exactly one target column exists only in train
    if len(candidates) == 1:
        return candidates[0]

    # If there are multiple, we can't safely infer
    return None


def _ensure_column(df: pd.DataFrame, col: str, default_value) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default_value
    return df


def _align_columns(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Make train/test have the same set of columns (order preserved with train as reference),
    adding missing columns with NaN.
    """
    train_cols = list(train.columns)
    test_cols = list(test.columns)

    all_cols = list(dict.fromkeys(train_cols + [c for c in test_cols if c not in train_cols]))

    for c in all_cols:
        if c not in train.columns:
            train[c] = pd.NA
        if c not in test.columns:
            test[c] = pd.NA

    # Reorder
    train = train[all_cols]
    test = test[all_cols]
    return train, test, all_cols


def _validate_split_column_absent(train: pd.DataFrame, test: pd.DataFrame):
    # Avoid accidentally overwriting a real column
    if SPLIT_COL in train.columns or SPLIT_COL in test.columns:
        raise ValueError(
            f"Column '{SPLIT_COL}' already exists in input. "
            f"Rename it in the source files or change SPLIT_COL constant."
        )


def _validate_id_column(train: pd.DataFrame, test: pd.DataFrame, id_col: Optional[str]):
    if not id_col:
        return
    if id_col not in train.columns:
        raise ValueError(f"--id '{id_col}' not found in train columns.")
    if id_col not in test.columns:
        raise ValueError(f"--id '{id_col}' not found in test columns.")


def main():
    parser = argparse.ArgumentParser(description="Combine Kaggle train/test CSVs into a single CSV with __split.")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--out", required=True, help="Output path for combined CSV")
    parser.add_argument("--id", dest="id_col", default=None, help="ID column name (optional but recommended)")
    parser.add_argument("--target", dest="target_col", default=None, help="Target/label column name (optional)")
    parser.add_argument("--drop-unknown-split", action="store_true",
                        help="If set, drops any rows whose __split would be unknown (not used here, kept for future).")
    args = parser.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    out_path = Path(args.out)

    train_df = _read_csv(train_path)
    test_df = _read_csv(test_path)

    _validate_split_column_absent(train_df, test_df)
    _validate_id_column(train_df, test_df, args.id_col)

    # Determine target column
    target_col = args.target_col
    if not target_col:
        inferred = _infer_target_column(train_df, test_df, args.id_col)
        if inferred:
            target_col = inferred
            print(f"[INFO] Inferred target column: {target_col}")
        else:
            print("[WARN] Could not infer target column automatically. "
                  "Pass --target <col> if you need the target explicitly included.")
            target_col = None

    # If target_col is known, ensure it exists on both
    if target_col:
        train_df = _ensure_column(train_df, target_col, pd.NA)  # should already exist
        test_df = _ensure_column(test_df, target_col, pd.NA)    # create NaN target for test

    # Align columns
    train_df, test_df, all_cols = _align_columns(train_df, test_df)

    # Add split column at the end (or you can put it first)
    train_df[SPLIT_COL] = TRAIN_TAG
    test_df[SPLIT_COL] = TEST_TAG

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # If id provided, ensure it's not missing
    if args.id_col:
        missing_id = combined[args.id_col].isna().sum()
        if missing_id > 0:
            print(f"[WARN] Combined dataset has {missing_id} missing values in id column '{args.id_col}'.")

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print("[OK] Combined CSV written.")
    print(f"     output: {out_path}")
    print(f"     rows: train={len(train_df)} test={len(test_df)} combined={len(combined)}")
    print(f"     columns: {len(combined.columns)} (includes '{SPLIT_COL}')")
    if target_col:
        n_train_target_missing = combined.loc[combined[SPLIT_COL] == TRAIN_TAG, target_col].isna().sum()
        n_test_target_missing = combined.loc[combined[SPLIT_COL] == TEST_TAG, target_col].isna().sum()
        print(f"     target: {target_col} (missing: train={n_train_target_missing}, test={n_test_target_missing})")
    else:
        print("     target: (not specified / not inferred)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
