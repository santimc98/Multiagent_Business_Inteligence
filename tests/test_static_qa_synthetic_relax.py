from src.agents.qa_reviewer import run_static_qa_checks


def test_static_qa_allows_aux_dataframe_from_row():
    code = """
import pandas as pd
df = pd.read_csv("data/cleaned_data.csv")
if df["age"].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
row = df.iloc[0].to_dict()
single = pd.DataFrame([row])
print("Mapping Summary:", {"target": "age", "features": ["age"]})
"""
    evaluation_spec = {
        "qa_gates": [{"name": "target_variance_guard", "severity": "HARD", "params": {}}],
        "canonical_columns": ["age"],
    }
    result = run_static_qa_checks(code, evaluation_spec)
    assert result is not None
    assert result.get("status") != "REJECTED"


def test_static_qa_rejects_literal_dataframe():
    code = """
import pandas as pd
df = pd.read_csv("data/cleaned_data.csv")
if df["age"].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
fake = pd.DataFrame({"a": [1, 2]})
print("Mapping Summary:", {"target": "age", "features": ["age"]})
"""
    evaluation_spec = {
        "qa_gates": [{"name": "no_synthetic_data", "severity": "HARD", "params": {}}],
        "canonical_columns": ["age"],
    }
    result = run_static_qa_checks(code, evaluation_spec)
    assert result is not None
    assert result.get("status") == "REJECTED"
    assert "no_synthetic_data" in (result.get("failed_gates") or [])


def test_static_qa_allows_randomforest_classifier():
    code = """
import pandas as pd
df = pd.read_csv("data/cleaned_data.csv")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
"""
    evaluation_spec = {"qa_gates": [{"name": "no_synthetic_data", "severity": "HARD", "params": {}}]}
    result = run_static_qa_checks(code, evaluation_spec)
    assert result is not None
    assert result.get("status") != "REJECTED"


def test_static_qa_rejects_numpy_random_calls():
    code = """
import pandas as pd
import numpy as np
pd.read_csv("data/cleaned_data.csv")
fake = np.random.rand(10)
"""
    evaluation_spec = {"qa_gates": [{"name": "no_synthetic_data", "severity": "HARD", "params": {}}]}
    result = run_static_qa_checks(code, evaluation_spec)
    assert result is not None
    assert result.get("status") == "REJECTED"
    assert "no_synthetic_data" in (result.get("failed_gates") or [])


def test_static_qa_allows_resampling_choice_when_param_enabled():
    code = """
import pandas as pd
import numpy as np
df = pd.read_csv("data/cleaned_data.csv")
idx = np.random.choice(len(df), size=len(df), replace=True)
boot = df.iloc[idx]
"""
    evaluation_spec = {
        "qa_gates": [
            {
                "name": "no_synthetic_data",
                "severity": "HARD",
                "params": {"allow_resampling_random": True},
            }
        ]
    }
    result = run_static_qa_checks(code, evaluation_spec)
    assert result is not None
    assert result.get("status") != "REJECTED"


def test_static_qa_relaxes_no_synth_for_metric_round_augmentation_hypothesis():
    code = """
import pandas as pd
import numpy as np
df = pd.read_csv("data/cleaned_data.csv")
noise = np.random.normal(0, 0.01, size=len(df))
df_aug = pd.DataFrame({"x": noise})
"""
    evaluation_spec = {
        "qa_gates": [{"name": "no_synthetic_data", "severity": "HARD", "params": {}}],
        "metric_improvement_round_active": True,
        "iteration_handoff": {
            "source": "actor_critic_metric_improvement",
            "hypothesis_packet": {
                "hypothesis": {"technique": "data_augmentation"}
            },
        },
    }
    result = run_static_qa_checks(code, evaluation_spec)
    assert result is not None
    assert result.get("status") != "REJECTED"
    warnings = result.get("warnings") or []
    assert any("NO_SYNTHETIC_DATA_RELAXED_FOR_AUGMENTATION" in str(item) for item in warnings)


def test_static_qa_relaxes_no_synth_when_gate_explicitly_allows_augmentation():
    code = """
import pandas as pd
import numpy as np
df = pd.read_csv("data/cleaned_data.csv")
fake = np.random.uniform(0, 1, size=len(df))
"""
    evaluation_spec = {
        "qa_gates": [
            {
                "name": "no_synthetic_data",
                "severity": "HARD",
                "params": {"allow_synthetic_augmentation": True},
            }
        ],
        "augmentation_requested": True,
    }
    result = run_static_qa_checks(code, evaluation_spec)
    assert result is not None
    assert result.get("status") != "REJECTED"
