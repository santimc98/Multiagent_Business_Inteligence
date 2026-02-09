import ast

from src.agents.data_engineer import DataEngineerAgent


def test_data_engineer_clean_code_recovers_from_trailing_reasoning_after_fence():
    agent = DataEngineerAgent.__new__(DataEngineerAgent)
    raw = """
# Decision Log:
# - deterministic cleanup
import pandas as pd
import numpy as np

df = pd.DataFrame({"x": [1, 2, 3]})
print(df.shape)
```
I need to double-check a few things:
1. this is reasoning text
The script looks good. I'll generate it now. </think># Decision Log:
# - another log line
"""
    cleaned = agent._clean_code(raw)
    ast.parse(cleaned)
    assert "import pandas as pd" in cleaned
    assert "I need to double-check" not in cleaned
