import json
import os

from src.agents.business_translator import BusinessTranslatorAgent


class _EchoModel:
    def generate_content(self, prompt):
        class R:
            text = prompt

        return R()


def test_translator_prompt_includes_report_narrative_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open("data/run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO", "status": "completed"}, f)
    with open("data/data_quality_shape_pack.json", "w", encoding="utf-8") as f:
        json.dump({"shape_signals": {"counts": {"high_missingness": 1}}}, f)
    with open("data/feature_governance_pack.json", "w", encoding="utf-8") as f:
        json.dump({"feature_governance_signals": {"semantic_duplicate_groups": [{"columns": ["a", "a_code"]}]}}, f)
    with open("data/integration_card.json", "w", encoding="utf-8") as f:
        json.dump({"input_contract": {"feature_count": 2}}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report({"execution_output": "OK", "business_objective": "Score customers."})

    assert "REPORT_NARRATIVE_CONTRACT" in prompt
    assert "data_quality_characterization" in prompt
    assert "feature_governance" in prompt
    assert "integration_readiness" in prompt
    assert "Use it as a coverage checklist" in prompt
