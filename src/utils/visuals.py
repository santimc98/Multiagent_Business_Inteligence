
import matplotlib
matplotlib.use('Agg') # Headless
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from typing import List

def generate_fallback_plots(
    csv_path: str,
    output_dir: str = "static/plots",
    sep: str = ",",
    decimal: str = ".",
    encoding: str = "utf-8",
) -> List[str]:
    """
    Generates minimal fallback plots (Histogram, Boxplot) if sandbox fails.
    Returns list of generated file paths.
    """
    generated_files = []
    
    if not os.path.exists(csv_path):
        print(f"Fallback Plotting Error: CSV not found at {csv_path}")
        return []
        
    try:
        df = pd.read_csv(csv_path, sep=sep, decimal=decimal, encoding=encoding)
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Identify Main Numeric Column (High Variance or just first numeric)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) == 0:
            print("Fallback: No numeric columns found.")
            return []
            
        main_col = numeric_cols[0]
        # Heuristic: Pick column with most unique values (likely continuous)
        max_unique = -1
        for col in numeric_cols:
            n = df[col].nunique()
            if n > max_unique:
                max_unique = n
                main_col = col
                
        # Plot 1: Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df[main_col], kde=True)
        plt.title(f"Distribution of {main_col} (Fallback)")
        plt.xlabel(main_col)
        hist_path = os.path.join(output_dir, "fallback_distribution.png")
        plt.savefig(hist_path, bbox_inches='tight')
        plt.close()
        generated_files.append(hist_path)
        
        # Plot 2: Boxplot by Region/Category
        # Heuristic: Look for 'region', 'country', 'segment' or first object column
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        group_col = None
        
        candidates = ['region', 'country', 'segment', 'category', 'class']
        for cand in candidates:
            match = next((c for c in cat_cols if cand in c.lower()), None)
            if match:
                group_col = match
                break
        
        if not group_col and len(cat_cols) > 0:
            # Pick first one with reasonably low cardinality (<20)
            for c in cat_cols:
                if df[c].nunique() < 20:
                    group_col = c
                    break
                    
        if group_col:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=group_col, y=main_col, data=df)
            plt.title(f"{main_col} by {group_col} (Fallback)")
            plt.xticks(rotation=45)
            box_path = os.path.join(output_dir, "fallback_boxplot.png")
            plt.savefig(box_path, bbox_inches='tight')
            plt.close()
            generated_files.append(box_path)
            
        print(f"Fallback Plots Generated: {generated_files}")
        return generated_files

    except Exception as e:
        print(f"Fallback Plotting Failed: {e}")
        return []
