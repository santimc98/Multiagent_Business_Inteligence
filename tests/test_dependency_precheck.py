from src.utils.sandbox_deps import check_dependency_precheck


def test_dependency_precheck_blocks_banned_import():
    code = "import pulp\n"
    result = check_dependency_precheck(code, required_dependencies=[])
    assert "pulp" in result["banned"]
    assert "pulp" in result.get("suggestions", {})


def test_dependency_precheck_allows_base_imports():
    code = "import pandas\nfrom sklearn.linear_model import LogisticRegression\nimport json\nimport statsmodels\n"
    result = check_dependency_precheck(code, required_dependencies=[])
    assert result["blocked"] == []
    assert result["banned"] == []


def test_dependency_precheck_allows_extended_when_contract_requests():
    code = "import rapidfuzz\n"
    result = check_dependency_precheck(code, required_dependencies=["rapidfuzz"])
    assert result["blocked"] == []
    assert result["banned"] == []


def test_dependency_precheck_blocks_extended_when_not_requested():
    code = "import rapidfuzz\n"
    result = check_dependency_precheck(code, required_dependencies=[])
    assert "rapidfuzz" in result["blocked"]
