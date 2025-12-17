import sys
import os
sys.path.append('.')
from src.agents.reviewer import ReviewerAgent

def test_reviewer_deterministic_triage():
    print("Testing Reviewer Deterministic Triage...")
    
    # Mock Reviewer (no API needed for deterministic path)
    reviewer = ReviewerAgent(api_key="mock")
    
    # Simulation 1: String to Float Error
    execution_implosion = """
    STDOUT:
    Loading data...
    Data loaded.
    
    STDERR:
    Traceback (most recent call last):
      File "script.py", line 45, in <module>
        df.corr()
      File "pandas/core/frame.py", line 1234, in corr
    ValueError: could not convert string to float: 'Male'
    """
    
    result = reviewer.evaluate_results(execution_implosion, "Predict Churn", "Strategy: Baseline")
    
    print(f"\n[Test 1] Input Error: 'could not convert string to float'")
    print(f"Status: {result['status']}")
    print(f"Feedback: {result['feedback']}")
    print(f"Required Fixes: {result['required_fixes']}")
    
    assert result['status'] == "NEEDS_IMPROVEMENT"
    assert any("pd.to_numeric" in fix for fix in result['required_fixes'])
    assert len(result['required_fixes']) >= 3
    print("✅ Test 1 Passed: Deterministic Triage caught the error.")

    # Simulation 2: Generic Traceback
    execution_generic = """
    EXECUTION ERROR:
    KeyError: 'target_col'
    """
    result2 = reviewer.evaluate_results(execution_generic, "Predict Churn", "Strategy: Baseline")
    
    print(f"\n[Test 2] Input Error: 'KeyError'")
    assert result2['status'] == "NEEDS_IMPROVEMENT"
    assert "Fix the Python Runtime Error" in result2['required_fixes'][0]
    print("✅ Test 2 Passed: Generic Traceback caught.")

if __name__ == "__main__":
    try:
        test_reviewer_deterministic_triage()
        print("\nALL REVIEWER TESTS PASSED")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
