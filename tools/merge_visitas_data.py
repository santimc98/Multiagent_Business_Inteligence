import pandas as pd
import os

def merge_data():
    # Get the directory where the script is located (tools/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root
    project_root = os.path.dirname(script_dir)
    
    file1_path = os.path.join(project_root, 'data', 'data (29).csv')
    file2_path = os.path.join(project_root, 'data', 'data (30).csv')
    output_path = os.path.join(project_root, 'data', 'data (29_30).csv')
    
    print(f"Loading {file1_path}...")
    df1 = pd.read_csv(file1_path, sep=';', decimal=',', encoding='utf-8-sig') # Assuming cleaned files are utf-8-sig or similar from previous steps? 
    # Wait, clean_visitas_data.py saved with default encoding (utf-8).
    # But let's be robust.
    
    print(f"Loading {file2_path}...")
    df2 = pd.read_csv(file2_path, sep=';', decimal=',', encoding='utf-8-sig') # clean_visitas_data.py might have saved as default utf-8.
    
    # Concatenate
    print("Concatenating...")
    df_merged = pd.concat([df1, df2], ignore_index=True)
    
    # Convert Date
    print("Converting dates...")
    # Using DateOf1stVisit as the primary date
    df_merged['DateOf1stVisit'] = pd.to_datetime(df_merged['DateOf1stVisit'], dayfirst=True, errors='coerce')
    
    # Sort
    print("Sorting...")
    df_merged = df_merged.sort_values(by='DateOf1stVisit')
    
    # Save
    print(f"Saving to {output_path}...")
    df_merged.to_csv(output_path, index=False, sep=';', decimal=',')
    print("Done!")

if __name__ == "__main__":
    merge_data()
