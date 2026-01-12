import json
import os

import pytest

from src.agents.cleaning_reviewer import CleaningReviewerAgent


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    old_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_cwd)


def test_cleaning_reviewer_uses_manifest_output_dialect(tmp_workdir):
    cleaned_path = tmp_workdir / "cleaned.csv"
    cleaned_path.write_text("alpha,beta,value\n1,2,3.5\n4,5,6.7\n", encoding="utf-8")
    manifest_path = tmp_workdir / "cleaning_manifest.json"
    manifest_path.write_text(
        json.dumps({"output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"}}),
        encoding="utf-8",
    )

    cleaning_view = {
        "required_columns": ["alpha", "beta", "value"],
        "dialect": {"sep": ";", "decimal": ",", "encoding": "utf-8"},
    }

    agent = CleaningReviewerAgent()
    result = agent.review_cleaning(
        cleaning_view,
        cleaned_csv_path=str(cleaned_path),
        cleaning_manifest_path=str(manifest_path),
    )

    assert result["status"] != "REJECTED"
    assert "required_columns_present" not in result.get("failed_checks", [])


def test_cleaning_reviewer_infers_cleaned_dialect_when_manifest_missing(tmp_workdir):
    cleaned_path = tmp_workdir / "cleaned.csv"
    cleaned_path.write_text("alpha,beta,value\n1,2,3.5\n4,5,6.7\n", encoding="utf-8")
    manifest_path = tmp_workdir / "cleaning_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    cleaning_view = {
        "required_columns": ["alpha", "beta", "value"],
        "dialect": {"sep": ";", "decimal": ",", "encoding": "utf-8"},
    }

    agent = CleaningReviewerAgent()
    result = agent.review_cleaning(
        cleaning_view,
        cleaned_csv_path=str(cleaned_path),
        cleaning_manifest_path=str(manifest_path),
    )

    warnings = " ".join(result.get("warnings", []))
    assert "DIALECT_AUTO_INFERRED_FOR_CLEANED" in warnings
    assert result["status"] != "REJECTED"
