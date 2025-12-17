import pandas as pd
import numpy as np
import random
import os

def generate_messy_data():
    np.random.seed(42)
    random.seed(42)
    
    n_rows = 500
    
    # 1. Generate Base Data
    data = {
        'customer_id': np.arange(1000, 1000 + n_rows),
        'age': np.random.randint(18, 80, size=n_rows),
        'plan_type': np.random.choice(['Basic', 'Pro', 'Enterprise'], size=n_rows),
        'monthly_spend': np.random.uniform(10, 200, size=n_rows),
        'last_login_days': np.random.randint(0, 30, size=n_rows),
        'support_calls': np.random.randint(0, 10, size=n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # 2. Apply Business Logic for Churn (Hidden Pattern)
    # Churn if support_calls > 5 AND monthly_spend > 100 (Expensive & Angry)
    def calculate_churn(row):
        prob = 0.1 # Base churn
        if row['support_calls'] > 5 and row['monthly_spend'] > 100:
            prob = 0.9
        elif row['support_calls'] > 7:
            prob = 0.6
        return 1 if random.random() < prob else 0

    df['churn'] = df.apply(calculate_churn, axis=1)
    
    # 3. Introduce Messiness
    
    # Duplicate IDs (5% of rows)
    n_dupes = int(n_rows * 0.05)
    dupe_indices = np.random.choice(df.index, size=n_dupes, replace=False)
    df.loc[dupe_indices, 'customer_id'] = df.loc[dupe_indices, 'customer_id'].sample(frac=1).values
    
    # Invalid Ages (2% of rows)
    n_bad_ages = int(n_rows * 0.02)
    bad_age_indices = np.random.choice(df.index, size=n_bad_ages, replace=False)
    df.loc[bad_age_indices, 'age'] = np.random.choice([-5, -1, 150, 200], size=n_bad_ages)
    
    # Typos in Plan Type (5% of rows)
    n_typos = int(n_rows * 0.05)
    typo_indices = np.random.choice(df.index, size=n_typos, replace=False)
    replacements = {'Basic': 'basic ', 'Pro': 'Pro_Plan', 'Enterprise': 'Enterprice'}
    df.loc[typo_indices, 'plan_type'] = df.loc[typo_indices, 'plan_type'].map(lambda x: replacements.get(x, x))
    
    # Nulls in Monthly Spend (3% of rows)
    n_nulls = int(n_rows * 0.03)
    null_indices = np.random.choice(df.index, size=n_nulls, replace=False)
    df.loc[null_indices, 'monthly_spend'] = np.nan
    
    # 4. Save Data
    os.makedirs('data', exist_ok=True)
    output_path = 'data/saas_messy_churn.csv'
    df.to_csv(output_path, index=False)
    print(f"Successfully generated messy data at: {output_path}")
    print(df.head())
    print("\nData Info:")
    print(df.info())

if __name__ == "__main__":
    generate_messy_data()
