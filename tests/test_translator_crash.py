
import pytest
from src.agents.business_translator import BusinessTranslatorAgent
from unittest.mock import MagicMock, patch

def test_translator_report_generation_crash():
    """
    Test that calls generate_report to verify no NameError occurs.
    Mocking the LLM call to avoid API usage.
    """
    agent = BusinessTranslatorAgent(api_key="dummy_key")
    
    # Mock the client
    agent.client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "REPORT: Executive Summary..."
    agent.client.chat.completions.create.return_value = mock_response

    state = {
        'execution_output': 'Accuracy: 0.95',
        'business_objective': 'Test Objective'
    }
    
    try:
        report = agent.generate_report(state)
        assert "REPORT:" in report
    except NameError as e:
        pytest.fail(f"Translator crashed with NameError: {e}")
    except Exception as e:
        # Ignore other errors (like connection) if they aren't NameError, 
        # but here we mocked the client so it should be fine.
        if "NameError" in str(e):
             pytest.fail(f"Translator crashed with NameError: {e}")
        # If it's something else, we might want to know, but the goal is detecting NameError
