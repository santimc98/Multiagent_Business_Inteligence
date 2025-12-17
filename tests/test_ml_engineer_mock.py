import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from agents.ml_engineer import MLEngineerAgent

def test_ml_engineer_mock():
    print("--- Starting Mock Test for MLEngineerAgent ---")
    
    # 1. Define Mock Inputs
    mock_strategy = {
        "title": "Predictive Maintenance",
        "hypothesis": "Vibration and temperature levels predict machine failure.",
        "required_columns": ["vibration", "temperature", "failure"],
        "reasoning": "We can use a Random Forest Classifier to identify patterns leading to failure."
    }
    mock_data_path = "data/sensor_data.csv"
    
    print(f"Mock Strategy: {mock_strategy['title']}")
    print(f"Mock Data Path: {mock_data_path}")
    print("-" * 30)

    try:
        # 2. Instantiate Agent
        agent = MLEngineerAgent()
        print("Agent initialized successfully.")

        # 3. Execute Generation
        print("Generating code... (this may take a few seconds)")
        code = agent.generate_code(mock_strategy, mock_data_path)

        # 4. Print Result
        print("\n--- Resulting Code ---")
        print(code)
        
        # 5. Validation
        if "```" in code:
            print("\n[WARNING] Markdown detected in output!")
        else:
            print("\n[SUCCESS] Output appears to be clean code.")
            
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")

if __name__ == "__main__":
    test_ml_engineer_mock()
