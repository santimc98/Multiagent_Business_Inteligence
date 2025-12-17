import re
import ast


def extract_code_block(text: str) -> str:
    """
    Extracts the first python/code fence block if present; otherwise returns trimmed text.
    """
    if not isinstance(text, str):
        return str(text)
    # ```python ... ```
    m = re.search(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # ``` ... ```
    m = re.search(r"```(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
