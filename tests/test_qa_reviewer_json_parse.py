from types import SimpleNamespace

from src.agents.qa_reviewer import QAReviewerAgent


def test_qa_reviewer_parses_wrapped_json_response() -> None:
    agent = QAReviewerAgent(api_key="dummy")
    agent.provider = "openai"

    wrapped_content = (
        "Analysis complete.\n"
        "```json\n"
        "{\n"
        '  "status": "APPROVED",\n'
        '  "feedback": "QA Passed",\n'
        '  "failed_gates": [],\n'
        '  "required_fixes": [],\n'
        '  "evidence": [{"claim":"ok","source":"missing"}]\n'
        "}\n"
        "```\n"
        "End."
    )

    class _DummyCompletions:
        @staticmethod
        def create(*_args, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=wrapped_content))]
            )

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=_DummyCompletions()))

    result = agent.review_code(
        code="import pandas as pd\ndf = pd.read_csv('data/cleaned_data.csv')\n",
        strategy={"title": "Test strategy"},
        business_objective="Test objective",
        evaluation_spec={"qa_gates": [{"name": "runtime_success", "severity": "SOFT"}]},
    )

    assert result.get("status") == "APPROVED"
    assert result.get("feedback") == "QA Passed"
    assert result.get("failed_gates") == []
