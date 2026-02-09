from src.agents.execution_planner import ExecutionPlannerAgent


def test_execution_planner_does_not_set_explicit_max_output_tokens(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", raising=False)
    agent = ExecutionPlannerAgent(api_key=None)

    assert "max_output_tokens" not in agent._generation_config


def test_execution_planner_generation_config_keeps_stable_defaults(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", raising=False)
    agent = ExecutionPlannerAgent(api_key=None)

    assert agent._generation_config["temperature"] == 0.0
    assert agent._generation_config["top_p"] == 0.9
    assert agent._generation_config["top_k"] == 40
