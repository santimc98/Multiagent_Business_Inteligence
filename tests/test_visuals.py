
import pytest
import os
import pandas as pd
from unittest.mock import MagicMock
from src.utils.visuals import generate_fallback_plots

def test_fallback_plots_generation(tmp_path):
    # Setup Data
    df = pd.DataFrame({
        'price': [10.5, 20.0, 15.5, 100.0, 50.5],
        'region': ['North', 'South', 'North', 'East', 'South'],
        'id': [1, 2, 3, 4, 5]
    })
    csv_file = tmp_path / "cleaned_data.csv"
    df.to_csv(csv_file, index=False)
    
    plots_dir = tmp_path / "plots"
    
    # Run
    generated = generate_fallback_plots(str(csv_file), output_dir=str(plots_dir))
    
    # Assert
    assert len(generated) >= 1
    assert any("fallback_distribution" in p for p in generated)
    assert any("fallback_boxplot" in p for p in generated)
    assert os.path.exists(generated[0])

def test_fallback_no_numeric(tmp_path):
    df = pd.DataFrame({'a': ['x', 'y'], 'b': ['m', 'n']})
    csv_file = tmp_path / "data.csv"
    df.to_csv(csv_file, index=False)
    
    generated = generate_fallback_plots(str(csv_file), output_dir=str(tmp_path))
    assert len(generated) == 0
