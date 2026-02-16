import os
import subprocess

files = [
    "cloudrun/heavy_runner/heavy_train.py",
    "src/agents/execution_planner.py", 
    "src/graph/graph.py", 
    "src/utils/contract_accessors.py", 
    "src/utils/contract_views.py"
]

os.makedirs("diffs_updates", exist_ok=True)

for f in files:
    clean_name = f.replace("/", "_").replace("\\", "_") + ".diff"
    dest = os.path.join("diffs_updates", clean_name)
    try:
        with open(dest, "w", encoding="utf-8") as outfile:
            subprocess.run(["git", "diff", "--", f], stdout=outfile, check=True)
        print(f"Created {dest}")
    except Exception as e:
        print(f"Error creating {dest}: {e}")
