import re
import ast


def extract_code_block(text: str) -> str:
    """
    Extracts the first python/code fence block if present; otherwise returns trimmed text.
    """
    if not isinstance(text, str):
        return str(text)
    # Prefer all fenced code blocks (python or generic) and join.
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    cleaned = [b.strip() for b in blocks if isinstance(b, str) and b.strip()]
    if cleaned:
        return "\n\n".join(cleaned).strip()
    # Handle unterminated fences (e.g., truncated output)
    m = re.search(r"```(?:python)?", text, re.IGNORECASE)
    if m:
        return text[m.end():].strip()
    return text.strip()


def is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
