from src.agents.ml_engineer import MLEngineerAgent


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, timeout=None, default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers


def test_editor_mode_prompt_includes_structured_feedback_json(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Test Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        feedback_history=["legacy reviewer text"],
        previous_code="print('previous')\n",
        gate_context={
            "source": "reviewer",
            "status": "REJECTED",
            "feedback": "legacy reviewer text",
            "feedback_json": {
                "version": "v1",
                "status": "REJECTED",
                "failed_gates": ["submission_format_validation"],
                "required_fixes": ["Write required outputs at exact paths."],
            },
            "failed_gates": ["submission_format_validation"],
            "required_fixes": ["Write required outputs at exact paths."],
        },
        iteration_handoff={"mode": "patch"},
        execution_contract={"required_outputs": ["data/metrics.json"], "canonical_columns": []},
        ml_view={"required_outputs": ["data/metrics.json"]},
        editor_mode=True,
    )

    prompt = str(agent.last_prompt or "")
    assert "LATEST_ITERATION_FEEDBACK_JSON" in prompt
    assert "submission_format_validation" in prompt
