import json

from src.agents.execution_planner import ExecutionPlannerAgent
from src.graph.graph import _resolve_required_input_columns, dialect_guard_violations, _filter_input_contract


def test_execution_planner_marks_missing_column_as_derived_with_inventory():
    planner = ExecutionPlannerAgent(api_key=None)
    strategy = {"required_columns": ["RefScore"], "title": "Ranking"}
    contract = planner.generate_contract(strategy=strategy, column_inventory=["col_a"])
    reqs = contract.get("data_requirements", [])
    ref_req = next((r for r in reqs if str(r.get("name")).lower() == "refscore"), {})
    assert ref_req.get("source") == "derived"


def test_resolve_required_input_columns_ignores_derived():
    contract = {
        "data_requirements": [
            {"name": "feature_a", "source": "input"},
            {"name": "target_x", "source": "derived"},
        ]
    }
    required = _resolve_required_input_columns(contract, {"required_columns": ["fallback"]})
    assert required == ["feature_a"]
    filtered = _filter_input_contract(contract)
    assert all(req.get("source") == "input" for req in filtered.get("data_requirements", []))


def test_dialect_guard_flags_mismatch():
    code = "import pandas as pd\npd.read_csv('data/raw.csv', sep=';', decimal=',', encoding='utf-8')\n"
    issues = dialect_guard_violations(code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8", expected_path="data/raw.csv")
    assert any("literal" in i or "missing" in i for i in issues)


def test_dialect_guard_accepts_matching_dialect():
    code = "import pandas as pd\nsep_val = ';'\npd.read_csv('data/raw.csv', sep=sep_val, decimal=',', encoding='utf-8')\n"
    issues = dialect_guard_violations(code, csv_sep=";", csv_decimal=",", csv_encoding="utf-8", expected_path="data/raw.csv")
    assert issues == []
