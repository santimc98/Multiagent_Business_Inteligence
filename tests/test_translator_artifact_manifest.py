import json
import os

from src.agents.business_translator import BusinessTranslatorAgent


class _EchoModel:
    def generate_content(self, prompt):
        class _Resp:
            def __init__(self, text):
                self.text = text

        return _Resp(prompt)


def test_translator_builds_artifact_manifest_and_html_tables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": 0.92}, f)

    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "required_outputs": ["data/metrics.json", "data/scored_rows.csv"],
                "artifact_requirements": {
                    "required_files": [{"path": "data/metrics.json"}, {"path": "data/scored_rows.csv"}]
                },
            },
            f,
        )

    with open(os.path.join("data", "output_contract_report.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_status": "error",
                "present": ["data/metrics.json"],
                "missing": ["data/scored_rows.csv"],
                "artifact_requirements_report": {
                    "status": "error",
                    "files_report": {
                        "present": ["data/metrics.json"],
                        "missing": ["data/scored_rows.csv"],
                    },
                },
            },
            f,
        )

    with open(os.path.join("data", "produced_artifact_index.json"), "w", encoding="utf-8") as f:
        json.dump([{"path": "data/metrics.json", "artifact_type": "metrics"}], f)

    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    report = agent.generate_report({"execution_output": "ok", "business_objective": "Objetivo"})

    assert os.path.exists(os.path.join("data", "report_artifact_manifest.json"))
    assert os.path.exists(os.path.join("data", "report_visual_tables.json"))

    with open(os.path.join("data", "report_artifact_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["summary"]["required_total"] >= 2
    assert manifest["summary"]["required_missing"] >= 1
    assert any(item.get("path") == "data/metrics.json" for item in manifest.get("items", []))

    assert "Artifact Inventory Table (HTML)" in report
    assert "artifact_inventory_table_html" not in report
    assert "exec-table artifact-inventory" in report


def test_translator_manifest_profiles_csv_dimensions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open(os.path.join("data", "scored_rows.csv"), "w", encoding="utf-8") as f:
        f.write("id,prediction,score\n1,1,0.93\n2,0,0.11\n")

    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump({"required_outputs": ["data/scored_rows.csv"]}, f)

    with open(os.path.join("data", "produced_artifact_index.json"), "w", encoding="utf-8") as f:
        json.dump([{"path": "data/scored_rows.csv", "artifact_type": "predictions"}], f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    _ = agent.generate_report({"execution_output": "ok", "business_objective": "Objetivo"})

    with open(os.path.join("data", "report_artifact_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scored = next(item for item in manifest["items"] if item["path"] == "data/scored_rows.csv")
    assert scored["present"] is True
    assert scored["row_count"] == 2
    assert scored["column_count"] == 3
