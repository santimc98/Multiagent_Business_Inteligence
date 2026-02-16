"""
Tests for MLEngineerAgent._check_training_policy_compliance.

NOTE (v5.0): Static AST/regex-based policy compliance was deprecated in favour of
execution-based validation.  The method now returns [] unconditionally.
These tests verify the deprecated method's API contract (returns empty list)
and document the original intent for future reference.
"""
import pytest

from src.agents.ml_engineer import MLEngineerAgent


def _agent():
    return MLEngineerAgent.__new__(MLEngineerAgent)


def test_training_policy_accepts_split_usage_when_plan_prefers_split():
    agent = _agent()
    code = """
import pandas as pd
df = pd.read_csv('data/cleaned_data.csv')
train_df = df[df['__split'] == 'train'].copy()
"""
    execution_contract = {"outcome_columns": ["SalePrice"]}
    ml_view = {}
    ml_plan = {
        "training_rows_policy": "only_rows_with_label",
        "split_column": "__split",
        "evidence_used": {
            "split_evaluation": "Used split column '__split' because it delineates train/test.",
            "split_candidates": [{"column": "__split", "values": ["train", "test"]}],
        },
    }
    issues = agent._check_training_policy_compliance(code, execution_contract, ml_view, ml_plan)
    assert issues == []


@pytest.mark.skip(reason="Deprecated v5.0: static training policy checks moved to execution-based QA validation")
def test_training_policy_flags_missing_filter_when_no_split_or_label():
    agent = _agent()
    code = "df = pd.read_csv('data/cleaned_data.csv')"
    execution_contract = {"outcome_columns": ["SalePrice"]}
    ml_view = {}
    ml_plan = {"training_rows_policy": "only_rows_with_label"}
    issues = agent._check_training_policy_compliance(code, execution_contract, ml_view, ml_plan)
    assert "training_rows_filter_missing" in issues


@pytest.mark.skip(reason="Deprecated v5.0: static training policy checks moved to execution-based QA validation")
def test_training_policy_infers_split_column_from_evidence():
    agent = _agent()
    code = "df = pd.read_csv('data/cleaned_data.csv')"
    execution_contract = {"outcome_columns": ["SalePrice"]}
    ml_view = {}
    ml_plan = {
        "training_rows_policy": "use_split_column",
        "evidence_used": {
            "split_candidates": [{"column": "__split", "values": ["train", "test"]}],
            "split_evaluation": "Use split column.",
        },
    }
    issues = agent._check_training_policy_compliance(code, execution_contract, ml_view, ml_plan)
    assert "split_column_filter_missing" in issues


@pytest.mark.skip(reason="Deprecated v5.0: static training policy checks moved to execution-based QA validation")
def test_training_policy_requires_label_filter_when_train_filter_explicit():
    agent = _agent()
    code = """
import pandas as pd
df = pd.read_csv('data/cleaned_data.csv')
train_df = df[df['__split'] == 'train'].copy()
"""
    execution_contract = {"outcome_columns": ["SalePrice"]}
    ml_view = {}
    ml_plan = {
        "training_rows_policy": "only_rows_with_label",
        "split_column": "__split",
        "train_filter": {"type": "label_not_null", "column": "SalePrice"},
    }
    issues = agent._check_training_policy_compliance(code, execution_contract, ml_view, ml_plan)
    assert "training_rows_filter_missing" in issues
