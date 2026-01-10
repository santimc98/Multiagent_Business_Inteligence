from src.graph.graph import manifest_dump_missing_default


def test_manifest_guard_flags_dump_without_default():
    code = """
import json
def save(m, f):
    json.dump(m, f, indent=2)
"""
    assert manifest_dump_missing_default(code) is True


def test_manifest_guard_allows_dump_with_default():
    code = """
import json
def save(m, f, fn):
    json.dump(m, f, indent=2, default=fn)
"""
    assert manifest_dump_missing_default(code) is False


def test_manifest_guard_allows_safe_dump_override():
    code = """
import json
def _safe_dump_json(obj, fp, **kwargs):
    return json.dump(obj, fp, **kwargs)
json.dump = _safe_dump_json
def save(m, f):
    json.dump(m, f)
"""
    assert manifest_dump_missing_default(code) is False
