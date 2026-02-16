import pytest
import os
import re

# We test the AGENT FILES themselves to ensure they explicitly ban dangerous APIs
# in their prompts, preventing the LLM from generating insecure code.

AGENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/agents'))

def test_pandas_private_api_ban_in_prompts():
    """
    Scans agent files to ensure they explicitly mention the BAN on private APIs
    or dangerous imports in their sandbox security section.
    """
    target_agents = ["data_engineer.py"]
    
    for filename in target_agents:
        path = os.path.join(AGENTS_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if the prompt contains security ban strings (old or new format)
        has_ban = (
            "pandas.io.*" in content
            or "pd.io.parsers" in content
            or "SANDBOX SECURITY" in content
            or "BLOCKED IMPORTS" in content
        )
        if not has_ban:
            pytest.fail(f"{filename} does not seem to explicitly ban dangerous imports in its prompt.")

if __name__ == "__main__":
    try:
        test_pandas_private_api_ban_in_prompts()
        print("Pandas Private API Ban Check Passed")
    except Exception as e:
        print(f"Failed: {e}")
