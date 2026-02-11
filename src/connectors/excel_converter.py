"""Convert Excel files (.xlsx / .xls) to CSV."""

from __future__ import annotations

import os
import pandas as pd


def convert_to_csv(
    excel_path: str,
    sheet_name: str | None = None,
    output_dir: str = "data",
) -> str:
    """Read an Excel file and write it as a UTF-8 CSV.

    Parameters
    ----------
    excel_path:
        Path to the ``.xlsx`` or ``.xls`` file.
    sheet_name:
        Specific sheet to convert.  When *None* and the workbook contains
        multiple sheets, the sheet with the most rows is selected
        automatically.
    output_dir:
        Directory where the resulting CSV is saved.

    Returns
    -------
    str
        Absolute path to the generated CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if sheet_name is None:
        # Peek at all sheets to pick the one with the most rows
        all_sheets = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
        if len(all_sheets) == 1:
            sheet_name = next(iter(all_sheets))
            df = all_sheets[sheet_name]
        else:
            best_sheet = max(all_sheets, key=lambda k: len(all_sheets[k]))
            sheet_name = best_sheet
            df = all_sheets[best_sheet]
    else:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")

    basename = os.path.splitext(os.path.basename(excel_path))[0]
    csv_filename = f"{basename}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    df.to_csv(csv_path, index=False, encoding="utf-8")

    return os.path.abspath(csv_path)
