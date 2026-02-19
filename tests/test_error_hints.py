from src.utils.error_hints import derive_repair_hints


def test_invalid_categorical_feature_type_hint():
    traceback_text = """
Traceback (most recent call last):
  File "ml_engineer.py", line 188, in <module>
    train_model(X_train, y_train)
RuntimeError: Invalid type for categorical feature[feature_idx=7]=1.0 : categorical features must be integer or string
"""
    hints = derive_repair_hints(traceback_text)
    assert len(hints) == 1
    hint = hints[0]
    assert "categ" in hint.lower()
    assert "string o Int64" in hint
    assert "evita floats" in hint


def test_max_two_hints_and_deterministic_order():
    error_text = """
RuntimeError: Invalid type for categorical feature[feature_idx=0]=0.0 : categorical features must be integer or string
NameError: name 'sep' is not defined
TypeError: Object of type bool_ is not JSON serializable
"""
    hints = derive_repair_hints(error_text)
    assert len(hints) == 2
    assert "categ" in hints[0].lower()
    assert "Define sep/decimal/encoding" in hints[1]
