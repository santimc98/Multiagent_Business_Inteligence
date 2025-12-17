import pandas as pd
try:
    df = pd.read_csv('data/visitas_2025_cleaned.csv', sep=';', decimal=',', encoding='utf-8-sig')
    with open('cols.txt', 'w', encoding='utf-8') as f:
        for col in df.columns:
            f.write(f"{col}\n")
    print("Columns saved to cols.txt")
except Exception as e:
    print(e)
