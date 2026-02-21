from src.agents.qa_reviewer import _parse_json_payload_with_trace
from src.agents.reviewer import ReviewerAgent
from src.agents.qa_reviewer import QAReviewerAgent
from src.utils.reviewer_response_schema import (
    build_qa_response_schema,
    build_reviewer_response_schema,
)


def test_reviewer_parse_json_payload_repairs_trailing_comma() -> None:
    agent = ReviewerAgent(api_key="dummy")
    payload = (
        "```json\n"
        "{\n"
        '  "status": "APPROVED",\n'
        '  "feedback": "ok",\n'
        '  "failed_gates": [],\n'
        '  "required_fixes": [],\n'
        "}\n"
        "```\n"
        "Extra trailing commentary."
    )
    parsed = agent._parse_json_payload(payload)
    assert parsed.get("status") == "APPROVED"
    trace = agent.last_json_parse_trace or {}
    assert trace.get("chosen_step")
    assert trace.get("actor") == "reviewer"


def test_qa_parse_json_payload_repairs_trailing_comma() -> None:
    payload = (
        "Result:\n"
        "```json\n"
        "{\n"
        '  "status": "APPROVE_WITH_WARNINGS",\n'
        '  "feedback": "qa ok",\n'
        '  "failed_gates": [],\n'
        '  "required_fixes": [],\n'
        "}\n"
        "```"
    )
    parsed, trace = _parse_json_payload_with_trace(payload)
    assert parsed.get("status") == "APPROVE_WITH_WARNINGS"
    assert trace.get("chosen_step")
    assert trace.get("actor") == "qa_reviewer"


def test_reviewer_parse_json_payload_uses_llm_repair_when_truncated() -> None:
    agent = ReviewerAgent(api_key="dummy")

    class _FakeGemini:
        def __init__(self):
            self.calls = []

        def generate_content(self, prompt, generation_config=None):
            self.calls.append({"prompt": prompt, "generation_config": generation_config})
            return type(
                "_Resp",
                (),
                {
                    "text": (
                        '{"status":"APPROVED","feedback":"ok","failed_gates":[],"required_fixes":[],"hard_failures":[],'
                        '"evidence":[],"improvement_suggestions":{"techniques":[],"no_further_improvement":false}}'
                    )
                },
            )()

    fake = _FakeGemini()
    agent.provider = "gemini"
    agent.client = fake
    schema = build_reviewer_response_schema(["runtime_success"])
    parsed = agent._parse_json_payload_with_llm_repair(
        '{"status":"APPROVED"',
        schema=schema,
        repair_label="review_code",
    )
    assert parsed.get("status") == "APPROVED"
    trace = agent.last_json_parse_trace or {}
    assert trace.get("repair_via_llm") is True
    assert len(fake.calls) >= 1


def test_qa_parse_json_payload_uses_llm_repair_when_truncated() -> None:
    agent = QAReviewerAgent(api_key="dummy")

    class _FakeGemini:
        def __init__(self):
            self.calls = []

        def generate_content(self, prompt, generation_config=None):
            self.calls.append({"prompt": prompt, "generation_config": generation_config})
            return type(
                "_Resp",
                (),
                {
                    "text": (
                        '{"status":"APPROVE_WITH_WARNINGS","feedback":"ok","failed_gates":[],"required_fixes":[],'
                        '"hard_failures":[],"evidence":[]}'
                    )
                },
            )()

    fake = _FakeGemini()
    agent.provider = "gemini"
    agent.client = fake
    parsed, trace = agent._parse_json_with_llm_repair(
        '{"status":"APPROVE_WITH_WARNINGS"',
        qa_gate_names=["security_sandbox"],
        repair_label="review_code",
    )
    assert parsed.get("status") == "APPROVE_WITH_WARNINGS"
    assert trace.get("repair_via_llm") is True
    assert isinstance(build_qa_response_schema(["security_sandbox"]), dict)
    assert len(fake.calls) >= 1
