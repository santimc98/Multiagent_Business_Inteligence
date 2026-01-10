import ast
import re
from typing import List


def data_engineer_preflight(code: str) -> List[str]:
    """
    Deterministic static checks for common Data Engineer pitfalls.
    Returns a list of issues; empty list means allow.
    """
    issues: List[str] = []
    try:
        tree = ast.parse(code)
    except Exception:
        return issues

    # Import guard
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                root = (alias.name or "").split(".")[0]
                if root == "sys":
                    issues.append("Do not import sys; remove sys import.")
                    break

    # sum(x.sum()) pattern guard
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sum":
            if node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Call):
                    func = arg0.func
                    if (isinstance(func, ast.Attribute) and func.attr == "sum") or (
                        isinstance(func, ast.Name) and func.id == "sum"
                    ):
                        issues.append("Avoid sum(x.sum()); use mask.mean() or mask.sum()/len(mask) for ratios.")
                        break
    # Guard against any()/all() applied to a single membership test
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"any", "all"}:
            if len(node.args) != 1:
                continue
            arg0 = node.args[0]
            if not isinstance(arg0, ast.Compare):
                continue
            if not any(isinstance(op, (ast.In, ast.NotIn)) for op in arg0.ops):
                continue
            if not arg0.comparators:
                continue
            comp = arg0.comparators[0]
            if isinstance(comp, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                issues.append(
                    "any()/all() expects an iterable; avoid passing a single membership test (it returns a bool)."
                )
                break
    # Guard against slicing None for actual_column in validation summaries
    if "actual_column" in code:
        for line in code.splitlines():
            if "actual_column" in line and "[:" in line:
                issues.append(
                    "Guard actual_column when printing: use actual = str(res.get('actual_column') or 'MISSING') before slicing."
                )
                break
    if "df[actual_col].dtype" in code or "df[actual_col].dtypes" in code:
        issues.append(
            "Guard duplicate column labels: assign series = df[actual_col]; if isinstance(series, pd.DataFrame) use series = series.iloc[:, 0] before accessing dtype."
        )

    # Guard against fragile numeric parsers that don't sanitize currency/letters before float conversion.
    # This is a common cause of "all-NaN" required columns when raw values include symbols (€, $, %, etc).
    try:
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "parse_numeric":
                continue
            segment = ast.get_source_segment(code, node) or ""
            if not segment:
                continue
            uses_float = "float(" in segment or "pd.to_numeric" in segment
            has_sanitizer = (
                "re.sub" in segment
                or ".translate(" in segment
                or "isdigit" in segment
                or "^[\\s\\-\\+]*[\\d,." in segment  # crude regex-like guard
            )
            if uses_float and not has_sanitizer:
                issues.append(
                    "parse_numeric should strip non-numeric symbols (currency/letters) before conversion (e.g., re.sub(r\"[^0-9,\\.\\-+()%\\s]\", \"\", s))."
                )
                break
    except Exception:
        pass

    # Guard against dict usage before init inside loops (stats[col]["x"] before stats[col] = {}).
    try:
        loop_stack: List[dict] = []
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            while loop_stack and indent <= loop_stack[-1]["indent"]:
                loop_stack.pop()
            loop_match = re.match(r"^\s*for\s+(\w+)\s+in\b", line)
            if loop_match:
                loop_stack.append({"indent": indent, "var": loop_match.group(1), "inited": set()})
                continue
            if not loop_stack:
                continue
            for loop in loop_stack:
                if indent <= loop["indent"]:
                    continue
                var = loop["var"]
                init_match = re.search(rf"(\w+)\s*\[\s*{var}\s*\]\s*=", line)
                if init_match:
                    loop["inited"].add(init_match.group(1))
                setdefault_match = re.search(rf"(\w+)\.setdefault\(\s*{var}\s*,", line)
                if setdefault_match:
                    loop["inited"].add(setdefault_match.group(1))
                use_match = re.search(rf"(\w+)\s*\[\s*{var}\s*\]\s*\[", line)
                if use_match and use_match.group(1) not in loop["inited"]:
                    issues.append(
                        "Initialize per-column dict entries before nested assignments (e.g., stats[col] = {} before stats[col]['x'])."
                    )
                    return issues
    except Exception:
        pass
    return issues
