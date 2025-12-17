import pandas as pd
import os

def clean_data():
    # Get the directory where the script is located (tools/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root
    project_root = os.path.dirname(script_dir)
    
    input_path = os.path.join(project_root, 'data', 'data (29_30).csv')
    output_path = os.path.join(project_root, 'data', 'data (29_30)_cleaned.csv')
    
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    print(f"Loading {input_path}...")
    
    # Try loading with European format first (semicolon separator) as discovered previously
    try:
        df = pd.read_csv(input_path, sep=';', decimal=',', encoding='latin-1')
        print("Loaded with European format (sep=';').")
    except Exception:
        try:
            df = pd.read_csv(input_path, sep=',', decimal='.', encoding='latin-1')
            print("Loaded with Standard format (sep=',').")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return

    # Sanitize columns just in case
    df.columns = df.columns.str.strip()
    
    if 'Account' not in df.columns:
        print("Error: Column 'Account' not found in dataset.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    initial_rows = len(df)
    print(f"Initial rows: {initial_rows}")
    
    # Drop duplicates based on 'Account', keeping the first occurrence
    df_cleaned = df.drop_duplicates(subset=['Account'], keep='first')
    
    final_rows = len(df_cleaned)
    removed_rows = initial_rows - final_rows
    
    print(f"Final rows: {final_rows}")
    print(f"Removed {removed_rows} duplicate accounts.")
    
    # Save cleaned data
    # Use the same format (semicolon) to maintain consistency if it was european
    df_cleaned.to_csv(output_path, sep=';', decimal=',', index=False, encoding='latin-1')
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    clean_data()
