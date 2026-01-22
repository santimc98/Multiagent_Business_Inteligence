
import pytest
from unittest.mock import MagicMock, patch
import json
from src.agents.ml_engineer import MLEngineerAgent
from src.graph.graph import run_qa_reviewer

class TestMLPlanGeneration:
    def setup_method(self):
        self.agent = MLEngineerAgent(api_key="dummy")
        # Mock _execute_llm_call to avoid API hits
        self.agent._execute_llm_call = MagicMock()

    def test_generate_ml_plan_normal_dict(self):
        """Test happy path with valid JSON dict."""
        valid_plan = {'training_rows_policy': 'use_all', 'metric_policy': {}, 'cv_policy': {}}
        self.agent._execute_llm_call.return_value = json.dumps(valid_plan)
        
        plan = self.agent.generate_ml_plan({}, business_objective="obj")
        assert plan['training_rows_policy'] == 'use_all'

    def test_generate_ml_plan_list_normalization(self):
        """Test normalization when LLM returns [dict]."""
        valid_plan = {'training_rows_policy': 'custom', 'metric_policy': {}, 'cv_policy': {}}
        self.agent._execute_llm_call.return_value = json.dumps([valid_plan])
        
        plan = self.agent.generate_ml_plan({}, business_objective="obj")
        assert plan['training_rows_policy'] == 'custom'

    def test_generate_ml_plan_invalid_then_valid(self):
        """Test retry logic on invalid JSON."""
        # First call raises error or returns garbage, second returns proper JSON
        valid_plan = {'training_rows_policy': 'retry_success', 'metric_policy': {}, 'cv_policy': {}}
        
        # We need to simulate _execute_llm_call behaviour for multiple calls
        self.agent._execute_llm_call.side_effect = ["GARBAGE TEXT", json.dumps(valid_plan)]
        
        plan = self.agent.generate_ml_plan({}, business_objective="obj")
        assert plan['training_rows_policy'] == 'retry_success'
        assert self.agent._execute_llm_call.call_count == 2

    def test_generate_ml_plan_total_failure(self):
        """Test fallback on failure."""
        self.agent._execute_llm_call.side_effect = Exception("API Error")
        plan = self.agent.generate_ml_plan({})
        assert plan["plan_source"] == "llm_error"

class TestQAReviewerWiring:
    @patch('src.graph.graph.collect_static_qa_facts', return_value={})
    @patch('src.graph.graph._abort_if_requested', return_value=None)
    @patch('src.graph.graph._consume_budget', return_value=(True, {}, ""))
    @patch('src.graph.graph.run_static_qa_checks')
    def test_qa_reviewer_receives_ml_plan(self, mock_qa_checks, mock_budget, mock_abort, mock_facts):
        """Verify ml_plan is passed to QA Reviewer via context."""
        
        mock_qa_checks.return_value = {"status": "APPROVED"}
        
        ml_plan = {"training_rows_policy": "TEST_POLICY"}
        profile = {"basic_stats": "TEST_STATS"}
        
        # Build state
        state = {
            "ml_plan": ml_plan,
            "data_profile": profile,
            "generated_code": "print('hello')",
            "execution_contract": {"evaluation_spec": {}}, # ensure we have insertion point
            "qa_view": {"some":"view"}, # triggers qa_context creation
            "feedback_history": []
        }
        
        # We need to ensure contract_views or qa_view is present to enter the block where we hook
        run_qa_reviewer(state)
        
        # Verify call args
        args, kwargs = mock_qa_checks.call_args
        # user code, context, facts
        code_arg = args[0]
        context_arg = args[1]
        
        assert context_arg["evaluation_spec"]["ml_plan"] == ml_plan
        assert context_arg["evaluation_spec"]["data_profile"] == profile
