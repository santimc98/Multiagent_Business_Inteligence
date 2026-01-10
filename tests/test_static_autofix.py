from src.graph.graph import _apply_static_autofixes
from src.utils.static_safety_scan import scan_code_safety


def test_auto_fix_np_bool_passes_static_scan():
    code = "import numpy as np\nflag = np.bool(1)\n"
    fixed, fixes = _apply_static_autofixes(code)
    assert "np.bool_" in fixed
    assert fixes
    is_safe, violations = scan_code_safety(fixed)
    assert is_safe
    assert not violations


def test_autofix_nested_dict_assignment():
    code = "stats = {}\nstats['A']['b'] = 1\n"
    fixed, fixes = _apply_static_autofixes(code)
    assert "stats.setdefault('A', {})['b']" in fixed
    assert any(fix.get("rule") == "nested_dict_setdefault" for fix in fixes)
