from src.agents.ml_engineer import MLEngineerAgent


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, timeout=None, default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers


def test_editor_mode_prompt_includes_repair_hints(monkeypatch):
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
    feedback_with_hints = (
        "Runtime error observed.\n\n"
        "REPAIR_HINTS (deterministic, no-autopatch):\n"
        "- Tipo invalido en columnas categoricas: convierte las categorias a string o Int64 antes de entrenar; evita floats (0.0/1.0) como categorias y pasa las columnas categoricas como lo requiera tu stack (por nombre/indice/selector)."
    )
    _ = agent.generate_code(
        strategy={"title": "Test Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        feedback_history=[feedback_with_hints],
        previous_code="print('previous')\n",
        gate_context={
            "source": "Execution Runtime",
            "status": "REJECTED",
            "feedback": feedback_with_hints,
            "failed_gates": ["runtime_failure"],
            "required_fixes": ["Fix runtime failure."],
        },
        iteration_handoff={
            "mode": "patch",
            "editor_constraints": {
                "must_apply_hypothesis": True,
                "forbid_noop": True,
                "patch_intensity": "aggressive",
            },
        },
        execution_contract={"required_outputs": ["data/metrics.json"], "canonical_columns": []},
        ml_view={"required_outputs": ["data/metrics.json"]},
        editor_mode=True,
    )

    assert "REPAIR_HINTS (deterministic, no-autopatch):" in str(agent.last_prompt or "")
    assert "Tipo invalido en columnas categoricas" in str(agent.last_prompt or "")
    assert "Metric-improvement round enforcement is ACTIVE." in str(agent.last_prompt or "")
