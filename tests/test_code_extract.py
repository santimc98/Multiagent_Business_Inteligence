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
