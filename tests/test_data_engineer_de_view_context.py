from types import SimpleNamespace
from unittest.mock import patch

from src.agents.data_engineer import DataEngineerAgent


def _mock_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_data_engineer_uses_de_view_for_gates_and_runbook_when_contract_missing():
    agent = DataEngineerAgent(api_key="fake")
    de_view = {
        "required_columns": ["id", "feature_a"],
        "optional_passthrough_columns": [],
        "output_path": "data/cleaned_data.csv",
        "output_manifest_path": "data/cleaning_manifest.json",
        "cleaning_gates": [
            {"name": "required_columns_present", "severity": "HARD", "params": {}},
        ],
        "data_engineer_runbook": {"steps": ["strict_schema", "preserve_labels"]},
    }

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('ok')"), "mock/model"),
    ):
        code = agent.generate_cleaning_script(
            data_audit="audit",
            strategy={"required_columns": ["id", "feature_a"]},
            input_path="data/raw.csv",
            de_view=de_view,
        )

    assert "print('ok')" in code
    prompt = agent.last_prompt or ""
    assert "required_columns_present" in prompt
    assert "strict_schema" in prompt
    assert '"output_path": "data/cleaned_data.csv"' in prompt
