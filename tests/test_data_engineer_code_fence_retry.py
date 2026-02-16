from src.graph import graph as graph_module
from src.utils.run_bundle import init_run_bundle


class StubDataEngineer:
    """Stub that returns code-fenced output on first call to test no-retry behavior."""
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
        # Return code-fenced output (markdown-wrapped code)
        # The system should NOT retry for code fences — it should strip & continue.
        self.last_response = "```python\nprint('clean')\n```"
        return "print('clean')"


def test_data_engineer_does_not_retry_on_code_fence(tmp_path, monkeypatch):
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
        "run_id": "testrun-code-fence",
        "run_start_epoch": 0,
    }
    run_dir = tmp_path / "runs" / state["run_id"]
    init_run_bundle(state["run_id"], state, run_dir=str(run_dir))

    stub = StubDataEngineer()
    monkeypatch.setattr(graph_module, "data_engineer", stub)

    graph_module.run_data_engineer(state)

    # Code fences should NOT cause a separate retry — the system strips them
    # and continues. The stub may be called more than once by other retry
    # loops (cleaning reviewer, etc.) but the key assertion is that
    # "CODE_FENCE_GUARD" never appears in the data_audit context.
    assert len(stub.calls) >= 1
    assert all("CODE_FENCE_GUARD" not in call for call in stub.calls)
