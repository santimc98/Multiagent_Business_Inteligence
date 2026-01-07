from src.graph.graph import _artifact_alignment_gate


def test_artifact_alignment_gate_allows_expected_revenue(tmp_path):
    cleaned_path = tmp_path / "cleaned.csv"
    scored_path = tmp_path / "scored.csv"
    cleaned_path.write_text(
        "CurrentPhase,1stYearAmount,Size,Debtors,Sector,Probability\n"
        "A,1,2,3,X,0.1\n"
        "B,2,3,4,Y,0.2\n",
        encoding="utf-8",
    )
    scored_path.write_text(
        "CurrentPhase,1stYearAmount,Size,Debtors,Sector,Probability,is_success,client_segment,success_probability,expected_revenue\n"
        "A,1,2,3,X,0.1,1,segA,0.8,100\n"
        "B,2,3,4,Y,0.2,0,segB,0.2,200\n",
        encoding="utf-8",
    )
    contract = {
        "required_outputs": ["data/scored_rows.csv"],
        "artifact_schemas": {
            "data/scored_rows.csv": {"allowed_extra_columns": ["expected_revenue"]}
        },
        "feature_engineering_plan": {"derived_columns": [{"name": "is_success"}]},
    }
    issues = _artifact_alignment_gate(
        str(cleaned_path),
        str(scored_path),
        contract,
        None,
        ",",
        ".",
        "utf-8",
    )
    assert not any(issue.startswith("scored_rows_unknown_columns") for issue in issues)


def test_artifact_alignment_gate_rejects_current_price(tmp_path):
    cleaned_path = tmp_path / "cleaned.csv"
    scored_path = tmp_path / "scored.csv"
    cleaned_path.write_text(
        "CurrentPhase,1stYearAmount,Size,Debtors,Sector,Probability\n"
        "A,1,2,3,X,0.1\n"
        "B,2,3,4,Y,0.2\n",
        encoding="utf-8",
    )
    scored_path.write_text(
        "CurrentPhase,1stYearAmount,Size,Debtors,Sector,Probability,current_price\n"
        "A,1,2,3,X,0.1,100\n"
        "B,2,3,4,Y,0.2,120\n",
        encoding="utf-8",
    )
    contract = {
        "required_outputs": ["data/scored_rows.csv"],
        "artifact_schemas": {
            "data/scored_rows.csv": {
                "allowed_name_patterns": [
                    r"^(expected|optimal|recommended)_.*(revenue|value|price|profit|margin|cost).*"
                ]
            }
        },
    }
    issues = _artifact_alignment_gate(
        str(cleaned_path),
        str(scored_path),
        contract,
        None,
        ",",
        ".",
        "utf-8",
    )
    assert any(issue.startswith("scored_rows_unknown_columns") for issue in issues)


def test_artifact_alignment_gate_rejects_age(tmp_path):
    cleaned_path = tmp_path / "cleaned.csv"
    scored_path = tmp_path / "scored.csv"
    cleaned_path.write_text(
        "CurrentPhase,1stYearAmount,Size,Debtors,Sector,Probability\n"
        "A,1,2,3,X,0.1\n"
        "B,2,3,4,Y,0.2\n",
        encoding="utf-8",
    )
    scored_path.write_text(
        "CurrentPhase,1stYearAmount,Size,Debtors,Sector,Probability,age\n"
        "A,1,2,3,X,0.1,30\n"
        "B,2,3,4,Y,0.2,45\n",
        encoding="utf-8",
    )
    contract = {"required_outputs": ["data/scored_rows.csv"]}
    issues = _artifact_alignment_gate(
        str(cleaned_path),
        str(scored_path),
        contract,
        None,
        ",",
        ".",
        "utf-8",
    )
    assert any(issue.startswith("scored_rows_unknown_columns") for issue in issues)
