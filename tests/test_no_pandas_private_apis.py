import pytest
import os
import re

# We test the AGENT FILES themselves to ensure they don't hallucinate these imports in their prompts/templates,
# AND we ideally would test generated code, but for now we static scan the agent source to ensure they ban them.

AGENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/agents'))

def test_pandas_private_api_ban_in_prompts():
    """
    Scans agent files to ensure they explicitly mention the BAN on private APIs.
    """
    target_agents = ["data_engineer.py"]
    
    for filename in target_agents:
        path = os.path.join(AGENTS_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if the prompt contains the ban string
        if "pandas.io.*" not in content and "pd.io.parsers" not in content:
            pytest.fail(f"{filename} does not seem to explicitly ban pandas.io private APIs in its prompt.")

if __name__ == "__main__":
    try:
        test_pandas_private_api_ban_in_prompts()
        print("Pandas Private API Ban Check Passed")
    except Exception as e:
        print(f"Failed: {e}")
