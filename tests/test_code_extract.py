import pytest

from src.utils.code_extract import extract_code_block, is_syntax_valid


def test_extract_code_block_from_fence():
    text = """
Here is code:
```python
print("hello")
```
"""
    extracted = extract_code_block(text)
    assert extracted.strip() == 'print("hello")'


def test_is_syntax_valid():
    assert is_syntax_valid("a=1")
    assert not is_syntax_valid("for")


def test_extract_code_block_prefers_prefix_when_single_closing_fence():
    text = """
import pandas as pd
df = pd.DataFrame({"a": [1, 2]})
print(df.shape)
```
I need to double-check a few things:
1. this is reasoning text, not code.
"""
    extracted = extract_code_block(text)
    assert "import pandas as pd" in extracted
    assert "I need to double-check" not in extracted


def test_extract_code_block_handles_open_unterminated_fence():
    text = """
Here is code:
```python
print("hello")
"""
    extracted = extract_code_block(text)
    assert extracted.strip() == 'print("hello")'
