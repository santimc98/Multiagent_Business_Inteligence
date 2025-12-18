from src.agents.execution_planner import ExecutionPlannerAgent


def test_execution_planner_expected_kind_from_summary():
    planner = ExecutionPlannerAgent(api_key=None)
    summary = (
        "Column Types:\n"
        "- Dates: fe\n"
        "- Numerical Features: fec\n"
        "- Categorical/Boolean: cat\n"
    )
    strategy = {"required_columns": ["fe", "fec", "cat"]}
    contract = planner.generate_contract(strategy=strategy, data_summary=summary, column_inventory=["fe", "fec", "cat"])
    reqs = {r["name"]: r for r in contract.get("data_requirements", [])}
    assert reqs["fe"]["expected_kind"] == "datetime"
    assert reqs["fec"]["expected_kind"] == "numeric"
    assert reqs["cat"]["expected_kind"] == "categorical"
