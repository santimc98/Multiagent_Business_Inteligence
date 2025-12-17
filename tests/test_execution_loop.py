
import pytest
from unittest.mock import MagicMock, patch
from src.graph.graph import check_execution_status, AgentState

def test_check_execution_status_retry():
    state = {
        "execution_output": "Traceback (most recent call last):\nValueError: bad",
        "execution_attempt": 1
    }
    assert check_execution_status(state) == "failed"

def test_check_execution_status_evaluate():
    state = {
        "execution_output": "Success",
        "execution_attempt": 1
    }
    assert check_execution_status(state) == "evaluate"

def test_check_execution_status_max_retries():
    state = {
        "execution_output": "Traceback...",
        "execution_attempt": 4
    }
    assert check_execution_status(state) == "evaluate"
