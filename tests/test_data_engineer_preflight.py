from src.utils.data_engineer_preflight import data_engineer_preflight


def test_data_engineer_preflight_blocks_double_sum():
    code = """
import pandas as pd
def main():
    s = pd.Series([1,2,3])
    ratio = sum(s.sum())
    return ratio
"""
    issues = data_engineer_preflight(code)
    assert any("sum" in i.lower() for i in issues)


def test_data_engineer_preflight_allows_mean_pattern():
    code = """
import pandas as pd
def main():
    s = pd.Series([True, False, True])
    ratio = s.mean()
    return ratio
"""
    issues = data_engineer_preflight(code)
    assert issues == []
