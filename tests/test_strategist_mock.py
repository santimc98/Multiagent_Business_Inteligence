import sys
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from agents.strategist import StrategistAgent

def test_strategist_mock():
    print("--- Starting Mock Test for StrategistAgent ---")
    
    # 1. Define Mock Data
    mock_summary = 'Dataset de ventas de telecomunicaciones. Columnas: customer_id, monthly_bill, total_usage, churn (0/1), contract_type'
    mock_context = 'El objetivo es reducir la fuga de clientes (churn) identificando a los usuarios en riesgo antes de que se vayan'
    
    print(f"Mock Summary: {mock_summary}")
    print(f"Mock Context: {mock_context}")
    print("-" * 30)

    try:
        # 2. Instantiate Agent
        # Note: This assumes GOOGLE_API_KEY is set in .env or environment
        agent = StrategistAgent()
        print("Agent initialized successfully.")

        # 3. Execute Generation
        print("Generating strategies... (this may take a few seconds)")
        strategies = agent.generate_strategies(mock_summary, mock_context)

        # 4. Print Result
        print("\n--- Resulting Strategies (JSON) ---")
        print(json.dumps(strategies, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        print("Tip: Ensure GOOGLE_API_KEY is set in your .env file.")

if __name__ == "__main__":
    test_strategist_mock()
