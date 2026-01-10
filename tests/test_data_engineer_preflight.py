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


def test_data_engineer_preflight_allows_init_before_nested():
    code = """
def main(cols):
    stats = {}
    for key in cols:
        stats[key] = {}
        stats[key]['x'] = 1
"""
    issues = data_engineer_preflight(code)
    assert issues == []


def test_data_engineer_preflight_allows_setdefault_nested():
    code = """
def main(cols):
    stats = {}
    for key in cols:
        stats.setdefault(key, {})['x'] = 1
"""
    issues = data_engineer_preflight(code)
    assert issues == []


def test_data_engineer_preflight_flags_nested_without_init():
    code = """
def main(cols):
    stats = {}
    for key in cols:
        stats[key]['x'] = 1
"""
    issues = data_engineer_preflight(code)
    assert any("Initialize per-column dict entries" in issue for issue in issues)


def test_data_engineer_preflight_flags_stats_shadowing():
    code = """
def main(cols):
    stats = {}
    for col in cols:
        stats.update({'a': col})
    stats['RIIM10 Norm'] = {}
    stats['RIIM10 Norm']['x'] = 1
"""
    issues = data_engineer_preflight(code)
    assert any("stats.update" in issue for issue in issues)
