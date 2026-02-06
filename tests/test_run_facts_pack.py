from src.utils.run_facts_pack import build_run_facts_pack


def test_run_facts_pack_prefers_contract_snapshot():
    state = {
        "run_id": "run_1",
        "csv_path": "data/input.csv",
        "execution_contract": {
            "contract_version": "4.1",
            "objective_type": "regression",
            "artifact_requirements": {
                "visual_requirements": {"enabled": False},
            },
        },
        "execution_contract_snapshot": {
            "execution_contract": {
                "contract_version": "4.2",
                "objective_type": "classification",
                "artifact_requirements": {
                    "visual_requirements": {"enabled": True, "required": True},
                },
            }
        },
        "execution_contract_source": "execution_planner_snapshot",
        "execution_contract_signature": "sig_full",
        "execution_contract_min_signature": "sig_min",
    }

    facts = build_run_facts_pack(state)
    assert facts["contract_source"] == "execution_planner_snapshot"
    assert facts["contract_signature"] == "sig_full"
    assert facts["contract_min_signature"] == "sig_min"
    assert facts["contract_version"] == "4.2"
    assert facts["objective_type"] == "classification"
    assert facts["visual_requirements"]["enabled"] is True


def test_run_facts_pack_falls_back_to_execution_contract():
    state = {
        "run_id": "run_2",
        "execution_contract": {
            "contract_version": "4.1",
            "objective_type": "ranking",
            "target_columns": ["target"],
            "artifact_requirements": {},
        },
    }

    facts = build_run_facts_pack(state)
    assert facts["contract_version"] == "4.1"
    assert facts["objective_type"] == "ranking"
    assert facts["target_columns"] == ["target"]
