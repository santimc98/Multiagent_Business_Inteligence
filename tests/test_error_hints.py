from src.utils.error_hints import derive_repair_hints


def test_catboost_invalid_cat_feature_hint():
    traceback_text = """
Traceback (most recent call last):
  File "ml_engineer.py", line 188, in <module>
    model.fit(X_train, y_train, cat_features=cat_features)
catboost.core.CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=7]=1.0 : cat_features must be integer or string
"""
    hints = derive_repair_hints(traceback_text)
    assert len(hints) == 1
    hint = hints[0]
    assert "CatBoost" in hint
    assert "string o Int64" in hint
    assert "no pases floats" in hint


def test_max_two_hints_and_deterministic_order():
    error_text = """
catboost.core.CatBoostError: Invalid type for cat_feature[0]=0.0
NameError: name 'sep' is not defined
TypeError: Object of type bool_ is not JSON serializable
"""
    hints = derive_repair_hints(error_text)
    assert len(hints) == 2
    assert "CatBoost" in hints[0]
    assert "Define sep/decimal/encoding" in hints[1]
