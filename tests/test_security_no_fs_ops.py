import pytest
import os

AGENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/agents'))

def test_security_fs_ops_ban_in_prompts():
    """
    Scans agent files to ensure they explicitly mention the BAN on FS OPs.
    """
    target_agents = ["ml_engineer.py", "data_engineer.py"]
    
    for filename in target_agents:
        path = os.path.join(AGENTS_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if the prompt contains the ban keyword
        if "NO NETWORK/FS OPS" not in content and "NO UNAUTHORIZED FS OPS" not in content:
             pytest.fail(f"{filename} does not seem to explicitly ban FileSystem/Network Ops in its prompt.")

if __name__ == "__main__":
    try:
        test_security_fs_ops_ban_in_prompts()
        print("Security FS Ban Check Passed")
    except Exception as e:
        print(f"Failed: {e}")
