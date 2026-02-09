import re
import ast
import json


def extract_code_block(text: str) -> str:
    """
    Extracts code-like content from LLM output.
    - If complete fenced blocks exist, returns their concatenation.
    - If an unterminated/single fence exists, chooses the most structured side
      (prefix/suffix) instead of always taking suffix.
    - Otherwise returns trimmed text.
    """
    if not isinstance(text, str):
        return str(text)
    # Prefer all fenced code blocks (python or generic) and join.
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    cleaned = [b.strip() for b in blocks if isinstance(b, str) and b.strip()]
    if cleaned:
        return "\n\n".join(cleaned).strip()
    # Handle unterminated / single fences (e.g., truncated output or stray fence).
    m = re.search(r"```(?:python)?", text, re.IGNORECASE)
    if m:
        prefix = text[:m.start()].strip()
        suffix = text[m.end():].strip()
        candidates = [c for c in (prefix, suffix) if isinstance(c, str) and c.strip()]
        if not candidates:
            return ""

        def _score(candidate: str) -> float:
            s = candidate.strip()
            score = 0.0
            if is_syntax_valid(s):
                score += 100.0
            try:
                json.loads(s)
                score += 90.0
            except Exception:
                pass
            lowered = s.lower()
            for marker in ("import ", "from ", "def ", "class ", "if __name__", " = ", "print(", "{", "}", "[", "]", "\""):
                score += 1.0 * lowered.count(marker)
            score += min(len(s), 12000) / 12000.0
            return score

        return max(candidates, key=_score)
    return text.strip()


def is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
