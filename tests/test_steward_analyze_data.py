import pandas as pd

from src.agents import steward as steward_module
from src.agents.steward import StewardAgent


def test_steward_analyze_data_shape_initialized(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "Size": [10, 20, 30],
            "Debtors": [1, 2, 3],
            "Sector": ["A", "B", "C"],
            "1stYearAmount": [100.0, 200.0, 300.0],
            "CurrentPhase": ["Prospect", "Negotiation", "Contract"],
            "Probability": [0.1, 0.5, 0.9],
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    class DummyResponse:
        text = "DATA SUMMARY: ok"

    class DummyModel:
        model_name = "dummy"

        def generate_content(self, _prompt):
            return DummyResponse()

    monkeypatch.setattr(steward_module.genai, "GenerativeModel", lambda *args, **kwargs: DummyModel())

    agent = StewardAgent(api_key="test-key")
    result = agent.analyze_data(str(csv_path), business_objective="test")
    summary = result.get("summary", "")
    assert "shape not associated" not in summary
