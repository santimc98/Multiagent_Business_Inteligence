import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from agents.strategist import StrategistAgent

def test_strategist_init():
    print("Testing StrategistAgent initialization...")
    # Mock API key for testing initialization logic only
    os.environ["GOOGLE_API_KEY"] = "dummy_key"
    
    try:
        agent = StrategistAgent()
        print("Successfully initialized StrategistAgent.")
        
        # Verify model config
        print(f"Model: {agent.model.model_name}")
        # Accessing generation_config might be different depending on SDK version, but basic init is done.
        
    except Exception as e:
        print(f"Failed to initialize: {e}")

if __name__ == "__main__":
    test_strategist_init()
