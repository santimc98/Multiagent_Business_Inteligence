from src.utils.integration_card import build_integration_card, summarize_integration_card


def test_integration_card_builds_input_execution_and_output_contract() -> None:
    contract = {
        "business_objective": "Score risk",
        "canonical_columns": ["feature_a", "feature_b"],
        "required_outputs": [
            {"path": "model.pkl", "intent": "model", "required": True},
            {"path": "scoring_function.py", "intent": "scoring_entrypoint", "required": True},
            {"path": "scored_rows.csv", "intent": "predictions", "required": True},
        ],
        "artifact_requirements": {
            "file_schemas": {
                "scored_rows.csv": {
                    "columns": [
                        {"name": "entity_id", "type": "string"},
                        {"name": "score", "type": "number"},
                    ]
                }
            }
        },
    }
    model_card = {
        "features_used": ["feature_a", "feature_b"],
        "model_size_bytes": 1234,
    }
    model_dep = {
        "model_dependency_signals": {
            "top_feature_share": 0.61,
            "top_source_family_share": 0.7,
            "top_features": [{"feature": "feature_a", "share": 0.61}],
        }
    }

    card = build_integration_card(
        contract=contract,
        model_card=model_card,
        model_dependency_context_pack=model_dep,
    )

    assert card["role"] == "integration_handoff_advisory"
    assert "must not reject" in card["deterministic_policy"]
    assert card["input_contract"]["feature_count"] == 2
    assert any(item["path"] == "scoring_function.py" for item in card["execution_contract"]["entrypoints"])
    scored = next(item for item in card["output_contract"]["artifacts"] if item["path"] == "scored_rows.csv")
    assert [col["name"] for col in scored["schema_columns"]] == ["entity_id", "score"]
    assert card["operational_dependency_context"]["top_feature_share"] == 0.61


def test_integration_card_summary_is_advisory() -> None:
    card = build_integration_card(
        contract={"required_outputs": ["predictions.csv"], "canonical_columns": ["x"]},
        metrics_payload={"features_used": ["x"]},
    )

    summary = summarize_integration_card(card)

    assert "INTEGRATION_CARD_SUMMARY" in summary
    assert "integration_handoff_advisory" in summary
    assert "not deterministic approval/rejection" in summary
    assert "feature_count: 1" in summary
