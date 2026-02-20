from src.agents.qa_reviewer import QAReviewerAgent


def test_qa_reviewer_generation_config_includes_response_schema_when_enabled(monkeypatch):
    monkeypatch.setenv("QA_REVIEWER_USE_RESPONSE_SCHEMA", "1")
    agent = QAReviewerAgent(api_key=None)

    cfg = agent._generation_config_for_review(["target_variance_guard", "security_sandbox"])

    assert "response_schema" in cfg
    schema = cfg.get("response_schema") or {}
    failed_items = (((schema.get("properties") or {}).get("failed_gates") or {}).get("items") or {})
    assert failed_items.get("enum") == ["target_variance_guard", "security_sandbox"]


def test_qa_reviewer_generate_gemini_retries_without_response_schema(monkeypatch):
    monkeypatch.setenv("QA_REVIEWER_USE_RESPONSE_SCHEMA", "1")
    agent = QAReviewerAgent(api_key=None)

    class _FakeGemini:
        def __init__(self):
            self.calls = []

        def generate_content(self, prompt, generation_config=None):
            self.calls.append({"prompt": prompt, "generation_config": generation_config})
            if isinstance(generation_config, dict) and "response_schema" in generation_config:
                raise ValueError("Unknown field response_schema")
            return type("_Resp", (), {"text": "{}"})()

    fake = _FakeGemini()
    agent.provider = "gemini"
    agent.client = fake

    content, used_config = agent._generate_gemini_json(
        "prompt",
        generation_config=agent._generation_config_for_review(["security_sandbox"]),
    )

    assert content == "{}"
    assert len(fake.calls) == 2
    first_cfg = fake.calls[0].get("generation_config") or {}
    second_cfg = fake.calls[1].get("generation_config") or {}
    assert "response_schema" in first_cfg
    assert "response_schema" not in second_cfg
    assert "response_schema" not in used_config
