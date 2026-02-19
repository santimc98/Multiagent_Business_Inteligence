from src.utils.contract_first_gates import apply_contract_first_gate_policy


def test_contract_first_gate_policy_downgrades_non_active_gates():
    active = ["required_artifacts_present"]
    packet = {
        "status": "REJECTED",
        "failed_gates": ["categorical_encoding_validation"],
        "hard_failures": ["categorical_encoding_validation"],
        "required_fixes": [
            {"gate": "categorical_encoding_validation", "instruction": "Fix categorical encoding."}
        ],
        "feedback": "Original feedback.",
    }

    result = apply_contract_first_gate_policy(packet, active, actor="qa_reviewer")

    assert result.get("failed_gates") == []
    assert result.get("hard_failures") == []
    assert result.get("required_fixes") == []
    assert result.get("status") in {"APPROVED", "APPROVE_WITH_WARNINGS"}
    assert "NON_ACTIVE_GATE_WARNINGS" in str(result.get("feedback") or "")
    assert "categorical_encoding_validation" in str(result.get("feedback") or "")


def test_contract_first_gate_policy_keeps_active_gate_blockers():
    active = ["categorical_encoding_validation"]
    packet = {
        "status": "REJECTED",
        "failed_gates": ["categorical_encoding_validation"],
        "hard_failures": ["categorical_encoding_validation"],
        "required_fixes": [
            {"gate": "categorical_encoding_validation", "instruction": "Fix categorical encoding."}
        ],
        "feedback": "Original feedback.",
    }

    result = apply_contract_first_gate_policy(packet, active, actor="qa_reviewer")

    assert result.get("failed_gates") == ["categorical_encoding_validation"]
    assert result.get("hard_failures") == ["categorical_encoding_validation"]
    assert result.get("status") == "REJECTED"


def test_contract_first_gate_policy_fail_closed_for_approved_with_active_hard_failure():
    active = ["categorical_encoding_validation"]
    packet = {
        "status": "APPROVED",
        "failed_gates": [],
        "hard_failures": ["categorical_encoding_validation"],
        "required_fixes": [],
        "feedback": "Looks good.",
    }

    result = apply_contract_first_gate_policy(packet, active, actor="reviewer")

    assert result.get("status") == "REJECTED"
    assert result.get("hard_failures") == ["categorical_encoding_validation"]
