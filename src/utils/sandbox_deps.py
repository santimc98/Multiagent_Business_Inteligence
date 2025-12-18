import ast
import sys
from typing import Iterable, Dict, List, Set

BASE_ALLOWLIST = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "openpyxl",
    "duckdb",
    "sqlalchemy",
    "dateutil",
    "pytz",
    "tqdm",
    "yaml",
]
EXTENDED_ALLOWLIST = ["rapidfuzz", "plotly", "pydantic", "pandera", "networkx"]
BANNED_ALLOWLIST = ["torch", "tensorflow", "pyspark", "spacy", "prophet", "cvxpy", "pulp", "fuzzywuzzy"]

PIP_BASE = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "openpyxl",
    "duckdb",
    "sqlalchemy",
    "python-dateutil",
    "pytz",
    "tqdm",
    "pyyaml",
]
PIP_EXTENDED = {
    "rapidfuzz": "rapidfuzz",
    "plotly": "plotly",
    "pydantic": "pydantic",
    "pandera": "pandera",
    "networkx": "networkx",
}


def _stdlib_modules() -> Set[str]:
    if hasattr(sys, "stdlib_module_names"):
        return set(sys.stdlib_module_names)
    return {
        "abc", "argparse", "asyncio", "base64", "collections", "contextlib", "csv",
        "dataclasses", "datetime", "enum", "functools", "glob", "hashlib", "itertools",
        "json", "logging", "math", "os", "pathlib", "random", "re", "statistics",
        "string", "sys", "time", "typing", "uuid", "warnings",
    }


def extract_import_roots(code: str) -> Set[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return set()
    roots: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def check_dependency_precheck(code: str, required_dependencies: Iterable[str] | None = None) -> Dict[str, List[str] | Dict[str, str]]:
    imports = extract_import_roots(code)
    required = {str(dep).split(".")[0] for dep in (required_dependencies or []) if dep}
    stdlib = _stdlib_modules()

    banned = sorted(imports & set(BANNED_ALLOWLIST))
    allowed = set(BASE_ALLOWLIST) | (set(EXTENDED_ALLOWLIST) & required) | stdlib
    blocked = sorted([imp for imp in imports if imp not in allowed])
    suggestions: Dict[str, str] = {}
    for imp in blocked + banned:
        if imp in {"pulp", "cvxpy"}:
            suggestions[imp] = "Use scipy.optimize.linprog or scipy.optimize.minimize (SLSQP)."
        elif imp in {"fuzzywuzzy"}:
            suggestions[imp] = "Use difflib or rapidfuzz (only if contract allows)."
        elif imp in {"rapidfuzz"}:
            suggestions[imp] = "Request rapidfuzz in execution_contract.required_dependencies, or use difflib."
        elif imp in {"torch", "tensorflow"}:
            suggestions[imp] = "Use scikit-learn or statsmodels."
        elif imp in {"spacy"}:
            suggestions[imp] = "Use standard NLP with scikit-learn or regex."
        elif imp in {"prophet"}:
            suggestions[imp] = "Use statsmodels or sklearn time-series approaches."
        elif imp in {"pyspark"}:
            suggestions[imp] = "Use pandas/pyarrow/duckdb for local processing."
    return {
        "imports": sorted(imports),
        "blocked": blocked,
        "banned": banned,
        "suggestions": suggestions,
    }


def get_sandbox_install_packages(required_dependencies: Iterable[str] | None = None) -> Dict[str, List[str]]:
    required = {str(dep).split(".")[0] for dep in (required_dependencies or []) if dep}
    extra = [PIP_EXTENDED[d] for d in EXTENDED_ALLOWLIST if d in required]
    return {"base": list(PIP_BASE), "extra": extra}
