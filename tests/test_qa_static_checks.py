import pytest

from src.agents.qa_reviewer import run_static_qa_checks


def test_static_qa_blocks_target_noise():
    code = """
import pandas as pd
import numpy as np
df = pd.read_csv('data/cleaned_data.csv')
y = df['Probability']
y = y + np.random.normal(0, 1, len(y))
"""
    evaluation_spec = {"qa_gates": [{"name": "target_variance_guard", "severity": "HARD", "params": {}}]}
    result = run_static_qa_checks(code, evaluation_spec=evaluation_spec)
    assert result is not None
    assert result["status"] == "REJECTED"
    assert "target_variance_guard" in result.get("failed_gates", [])


def test_static_qa_blocks_missing_variance_guard():
    code = """
import pandas as pd
df = pd.read_csv('data/cleaned_data.csv')
_ = df['Size']
print(df.shape)
"""
    evaluation_spec = {"qa_gates": [{"name": "target_variance_guard", "severity": "HARD", "params": {}}]}
    result = run_static_qa_checks(code, evaluation_spec=evaluation_spec)
    assert result is not None
    assert result["status"] == "REJECTED"
    assert "target_variance_guard" in result.get("failed_gates", [])


def test_static_qa_allows_variance_guard():
    code = """
import pandas as pd
df = pd.read_csv('data/cleaned_data.csv')
if df['Probability'].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
"""
    evaluation_spec = {"qa_gates": [{"name": "target_variance_guard", "severity": "HARD", "params": {}}]}
    result = run_static_qa_checks(code, evaluation_spec=evaluation_spec)
    assert result is not None
    assert result["status"] in {"PASS", "WARN"}


def test_static_qa_blocks_split_fabrication():
    code = """
import pandas as pd
df = pd.read_csv('data/cleaned_data.csv')
_ = df['Size']
df[['a','b']] = df['raw'].str.split(';', expand=True)
"""
    evaluation_spec = {"qa_gates": [{"name": "dialect_mismatch_handling", "severity": "HARD", "params": {}}]}
    result = run_static_qa_checks(code, evaluation_spec=evaluation_spec)
    assert result is not None
    assert "dialect_mismatch_handling" in result.get("failed_gates", [])


def test_static_qa_blocks_missing_group_split_when_inferred():
    code = """
import pandas as pd
from src.utils.group_split import infer_group_key
df = pd.read_csv('data/cleaned_data.csv')
_ = df['Sector']
if df['Probability'].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
groups = infer_group_key(df, exclude_cols=['target'])
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
for tr, te in kf.split(df):
    pass
"""
    evaluation_spec = {"qa_gates": [{"name": "group_split_required", "severity": "HARD", "params": {}}]}
    result = run_static_qa_checks(code, evaluation_spec=evaluation_spec)
    assert result is not None
    assert "group_split_required" in result.get("failed_gates", [])


def test_static_qa_allows_group_split_when_used():
    code = """
import pandas as pd
from src.utils.group_split import infer_group_key
from sklearn.model_selection import GroupKFold
df = pd.read_csv('data/cleaned_data.csv')
_ = df['Sector']
if df['Probability'].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
groups = infer_group_key(df, exclude_cols=['target'])
gkf = GroupKFold(n_splits=3)
for tr, te in gkf.split(df, df['target'], groups):
    pass
"""
    evaluation_spec = {"qa_gates": [{"name": "group_split_required", "severity": "HARD", "params": {}}]}
    result = run_static_qa_checks(code, evaluation_spec=evaluation_spec)
    assert result is not None
    assert result["status"] in {"PASS", "WARN"}


def test_static_qa_respects_explicit_gates_only():
    code = """
import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('data/cleaned_data.csv')
_ = df['CurrentPhase']
if df['Probability'].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
model = LinearRegression()
X = df[['a', 'b']]
y = df['target']
model.fit(X, y)
"""
    evaluation_spec = {"qa_gates": [{"name": "target_variance_guard", "severity": "HARD", "params": {}}]}
    result = run_static_qa_checks(code, evaluation_spec=evaluation_spec)
    assert result is not None
    assert result["status"] in {"PASS", "WARN"}


def test_static_qa_soft_gate_warns_only():
    code = """
import numpy as np
fake = np.random.rand(10)
"""
    evaluation_spec = {"qa_gates": [{"name": "no_synthetic_data", "severity": "SOFT", "params": {}}]}
    result = run_static_qa_checks(code, evaluation_spec=evaluation_spec)
    assert result is not None
    assert result["status"] == "WARN"
    assert "no_synthetic_data" in (result.get("soft_failures") or [])
    assert not (result.get("hard_failures") or [])


def test_static_qa_missing_gates_triggers_fallback_warning():
    code = "print('ok')"
    result = run_static_qa_checks(code, evaluation_spec={})
    assert result is not None
    assert result.get("contract_source_used") == "fallback"
    warnings = result.get("warnings") or []
    assert any("CONTRACT_BROKEN_FALLBACK" in w for w in warnings)
