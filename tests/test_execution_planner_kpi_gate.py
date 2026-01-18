from src.agents.execution_planner import _ensure_benchmark_kpi_gate


def test_ensure_benchmark_kpi_gate_inserts_metric_report():
    contract = {"qa_gates": [], "validation_requirements": {"method": "holdout"}}
    strategy = {"success_metric": "Accuracy"}
    updated = _ensure_benchmark_kpi_gate(contract, strategy, "")

    gates = updated.get("qa_gates", [])
    assert any(g.get("name") == "benchmark_kpi_report" for g in gates)
    gate = next(g for g in gates if g.get("name") == "benchmark_kpi_report")
    assert gate.get("params", {}).get("metric") == "accuracy"

    validation = updated.get("validation_requirements", {})
    assert validation.get("primary_metric") == "accuracy"
    assert "accuracy" in validation.get("metrics_to_report", [])
