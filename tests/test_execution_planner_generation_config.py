from src.agents.execution_planner import ExecutionPlannerAgent


def test_execution_planner_sets_max_output_tokens_floor(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", raising=False)
    agent = ExecutionPlannerAgent(api_key=None)

    assert int(agent._generation_config.get("max_output_tokens", 0)) >= 4000


def test_execution_planner_generation_config_keeps_stable_defaults(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", raising=False)
    agent = ExecutionPlannerAgent(api_key=None)

    assert agent._generation_config["temperature"] == 0.0
    assert agent._generation_config["top_p"] == 0.9
    assert agent._generation_config["top_k"] == 40


def test_execution_planner_max_output_tokens_env_below_floor_is_clamped(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", "1024")
    agent = ExecutionPlannerAgent(api_key=None)

    assert int(agent._generation_config.get("max_output_tokens", 0)) == 4000


def test_execution_planner_dynamic_budget_respects_context_window(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_CONTEXT_WINDOW_TOKENS", "9000")
    monkeypatch.setenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", "5000")
    agent = ExecutionPlannerAgent(api_key=None)
    prompt = "x" * 24000  # ~6000 prompt tokens with current estimator

    cfg = agent._generation_config_for_prompt(prompt)

    assert int(cfg.get("max_output_tokens", 0)) >= 1024
    assert int(cfg.get("max_output_tokens", 0)) <= int(agent._generation_config.get("max_output_tokens", 0))
