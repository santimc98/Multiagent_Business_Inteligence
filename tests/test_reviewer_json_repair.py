from src.agents.qa_reviewer import _parse_json_payload_with_trace
from src.agents.reviewer import ReviewerAgent


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
