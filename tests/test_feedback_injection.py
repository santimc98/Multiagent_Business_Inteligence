from src.utils.error_hints import append_repair_hints


def test_feedback_injection_appends_repair_hints_block():
    feedback = "FEEDBACK_BASE"
    error_text = """
Traceback (most recent call last):
catboost.core.CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=9]=1.0
"""
    new_feedback, hints = append_repair_hints(feedback, error_text)
    assert "FEEDBACK_BASE" in new_feedback
    assert "REPAIR_HINTS (deterministic, no-autopatch):" in new_feedback
    assert hints
    assert "CatBoost" in new_feedback
