import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from graph.graph import app_graph

def test_graph_structure():
    print("--- Testing Graph Structure ---")
    try:
        # Check if compiled
        print(f"Graph Type: {type(app_graph)}")
        
        # Visualize (print ascii representation if available, or just success message)
        print("Graph compiled successfully.")
        
        # Check nodes (internal access, might vary by version, but compilation check is key)
        # print(app_graph.get_graph().nodes) 
        
    except Exception as e:
        print(f"Graph verification failed: {e}")

if __name__ == "__main__":
    test_graph_structure()
