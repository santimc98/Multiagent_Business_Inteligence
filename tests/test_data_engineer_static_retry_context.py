from src.graph import graph as graph_module
from src.utils.run_bundle import init_run_bundle


class StubDataEngineer:
    """Stub that triggers a static scan failure on first call (forbidden import),
    then returns a valid script on the second call."""
    def __init__(self) -> None:
        self.calls = []
        self.model_name = "stub"
        self.last_prompt = None
        self.last_response = None

    def generate_cleaning_script(
        self,
        data_audit,
        strategy,
        input_path,
        business_objective="",
        csv_encoding="utf-8",
        csv_sep=",",
        csv_decimal=".",
        execution_contract=None,
        de_view=None,
        repair_mode=False,
    ) -> str:
        self.calls.append(data_audit)
        if len(self.calls) == 1:
            # This import triggers `scan_code_safety` to fail → retry
            return "import requests\nprint('hi')\n"
        # Second call: return valid code
        return "print('clean')"


def test_static_scan_retry_includes_violation_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    state = {
        "csv_path": str(csv_path),
        "selected_strategy": {"required_columns": []},
        "business_objective": "test",
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "data_summary": "summary",
        "execution_contract": {"required_outputs": ["data/cleaned_data.csv"]},
        "de_view": {
            "output_path": "data/cleaned_data.csv",
            "output_manifest_path": "data/cleaning_manifest.json",
            "required_columns": [],
            "cleaning_gates": [],
            "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        },
        "run_id": "testrun-static-scan",
        "run_start_epoch": 0,
    }
    run_dir = tmp_path / "runs" / state["run_id"]
    init_run_bundle(state["run_id"], state, run_dir=str(run_dir))

    stub = StubDataEngineer()
    monkeypatch.setattr(graph_module, "data_engineer", stub)

    graph_module.run_data_engineer(state)

    # The stub should be called at least twice: first with a forbidden import
    # (triggers static scan retry), then with valid code.
    assert len(stub.calls) >= 2
    # The retry context (second call) should mention the violation
    retry_context = stub.calls[1]
    assert "STATIC_SCAN" in retry_context or "ITERATION_FEEDBACK" in retry_context or "requests" in retry_context
