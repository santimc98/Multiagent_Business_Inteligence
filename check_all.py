import os
import sys
import tests.test_prompt_safety as t

SOURCE_DIR = os.path.join(os.path.dirname(__file__), 'src/agents')
files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.py')]

for f in files:
    path = os.path.join(SOURCE_DIR, f)
    try:
        if t.test_no_suspicious_fstrings(path):
            print(f"PASS: {f}")
        else:
            print(f"FAIL: {f}")
    except Exception as e:
        print(f"ERROR: {f} - {e}")
