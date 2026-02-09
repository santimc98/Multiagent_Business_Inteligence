from src.agents.execution_planner import ExecutionPlannerAgent


def test_execution_planner_sets_explicit_max_output_tokens(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", "20000")
    agent = ExecutionPlannerAgent(api_key=None)

    assert int(agent._generation_config.get("max_output_tokens", 0)) == 20000


def test_execution_planner_max_output_tokens_has_safe_floor(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", "10")
    agent = ExecutionPlannerAgent(api_key=None)

    assert int(agent._generation_config.get("max_output_tokens", 0)) >= 1024
