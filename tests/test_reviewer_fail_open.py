from src.agents.reviewer import ReviewerAgent


def test_reviewer_fail_open(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "test-key")
    reviewer = ReviewerAgent(api_key="test-key")

    class DummyCompletions:
        def create(self, *args, **kwargs):
            raise RuntimeError("boom")

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        chat = DummyChat()

    reviewer.client = DummyClient()
    result = reviewer.review_code("print('ok')")
    assert result.get("status") == "APPROVE_WITH_WARNINGS"
    assert "Reviewer unavailable" in result.get("feedback", "")
