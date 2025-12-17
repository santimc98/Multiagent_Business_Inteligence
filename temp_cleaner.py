import pandas as pd
import numpy as np
import os
import warnings
import re
import csv

warnings.filterwarnings('ignore')
input_path = r'data\data (29_30)_cleaned.csv'
output_path = 'data/cleaned_data.csv'

try:
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(1024)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            sep = dialect.delimiter
        except:
            sep = ','
    
    try:
        df = pd.read_csv(input_path, sep=sep, encoding='utf-8')
    except pd.errors.ParserError:
        for alt_sep in [';', '|', '\t']:
            try:
                df = pd.read_csv(input_path, sep=alt_sep, encoding='utf-8')
                break
            except:
                continue
        else:
            try:
                df = pd.read_csv(input_path, sep=sep, encoding='utf-8', engine='python')
            except:
                df = pd.read_csv(input_path, sep=sep, encoding='utf-8', engine='python', on_bad_lines='skip')
    
    initial_row_count = len(df)
    print(f"DEBUG: Initial Row Count: {initial_row_count}")
    
except Exception as e:
    print(f"CRITICAL: Failed to load CSV: {e}")
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            print("DEBUG: First 5 lines of file:")
            for i, line in enumerate(lines[:5]):
                print(f"Line {i+1}: {repr(line)}")
    except Exception as e2:
        print(f"DEBUG: Could not read file lines: {e2}")
    raise

def to_snake_case(name):
    name = str(name)
    name = re.sub(r'[^a-zA-Z0-9]+', '_', name)
    return name.strip('_').lower()

df.columns = [to_snake_case(col) for col in df.columns]
duplicates = df.columns[df.columns.duplicated()].tolist()
if duplicates:
    cols = pd.Series(df.columns)
    for dup in duplicates:
        mask = cols == dup
        count = mask.sum()
        for i in range(1, count):
            cols[mask.idxmax()] = f"{dup}_{i}"
    df.columns = cols

required_columns_original = ['CurrentPhase', '1stYearAmount', 'Size', 'Debtors', 'Sector', 'OppStatus', 'Probability', 'Country', 'SalesRep', 'Currency', 'LeadCost', 'VisitsCost', 'ConsultancyFee', 'AnnualSubscriptionFee', 'DateOfContract', 'DateOfClose', 'Reason', 'Account', 'Channel', 'ERP', 'Campaign']

def normalize_col(name):
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()

df_col_map = {normalize_col(c): c for c in df.columns}
final_cols_to_keep = []

for req_col in required_columns_original:
    norm_req = normalize_col(req_col)
    if norm_req in df_col_map:
        final_cols_to_keep.append(df_col_map[norm_req])
    else:
        print(f"WARNING: Required column '{req_col}' (norm: {norm_req}) not found in dataset.")

if final_cols_to_keep:
    missing_cols = [col for col in final_cols_to_keep if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Some columns not found after filtering: {missing_cols}")
        final_cols_to_keep = [col for col in final_cols_to_keep if col in df.columns]
    
    if final_cols_to_keep:
        df = df[final_cols_to_keep]
    else:
        print("CRITICAL: No columns to keep after filtering validation. Keeping all columns.")
else:
    print("CRITICAL: No matching columns found! Keeping original dataframe to avoid crash.")

df_before_drop = df.copy()
cols_to_drop = [col for col in df.columns if (df[col].isna().all() or df[col].nunique() <= 1) and col not in final_cols_to_keep]
df = df.drop(columns=cols_to_drop)
dropped_cols = set(df_before_drop.columns) - set(df.columns)
if dropped_cols:
    print(f"INFO: Dropped columns with no variance or all nulls: {dropped_cols}")

def clean_currency_column(series):
    if series.dtype == 'object':
        series = series.astype(str).str.replace(r'[€$£\s]', '', regex=True)
        mask_comma = series.str.contains(',')
        series.loc[mask_comma] = series.loc[mask_comma].str.replace('\.', '', regex=True).str.replace(',', '.')
        mask_multi_dot = series.str.count('\.') > 1
        def fix_multi_dots(x):
            if '.' in x:
                parts = x.split('.')
                return ''.join(parts[:-1]) + '.' + parts[-1]
            return x
        series.loc[mask_multi_dot] = series.loc[mask_multi_dot].apply(fix_multi_dots)
        series = pd.to_numeric(series, errors='coerce')
    return series

def clean_date_column(series):
    if series.dtype == 'object':
        series = pd.to_datetime(series, errors='coerce', dayfirst=True)
    return series

target_col = None
for col in df.columns:
    if normalize_col(col) == '1styearamount':
        target_col = col
        break

if not target_col:
    raise ValueError("CRITICAL: Target Variable '1stYearAmount' is missing from dataset")

currency_cols = []
for col in df.columns:
    norm_col = normalize_col(col)
    if norm_col in ['1styearamount', 'size', 'debtors', 'leadcost', 'visitcost', 'consultancyfee', 'annualsubscriptionfee']:
        currency_cols.append(col)

for col in currency_cols:
    if col in df.columns:
        df[col] = clean_currency_column(df[col])

prob_col = None
for col in df.columns:
    if normalize_col(col) == 'probability':
        prob_col = col
        break

if prob_col and prob_col in df.columns:
    df[prob_col] = df[prob_col].astype(str).str.rstrip('%')
    df[prob_col] = pd.to_numeric(df[prob_col], errors='coerce')
    if df[prob_col].max() > 1:
        df[prob_col] = df[prob_col] / 100

date_cols = []
for col in df.columns:
    norm_col = normalize_col(col)
    if norm_col in ['dateofcontract', 'dateofclose']:
        date_cols.append(col)

for col in date_cols:
    if col in df.columns:
        df[col] = clean_date_column(df[col])

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[target_col])

for col in df.columns:
    if col == target_col:
        continue
    
    if pd.api.types.is_numeric_dtype(df[col]):
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    elif df[col].dtype == 'object':
        if not df[col].isna().all():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)

df = df.dropna(how='all')
retention_rate = len(df) / initial_row_count if initial_row_count > 0 else 0
if retention_rate < 0.1:
    raise ValueError("CRITICAL: Aggressive cleaning detected. Dropped >90% of rows.")

if df[target_col].count() == 0:
    raise ValueError("CRITICAL: Target variable has insufficient data (empty)")
if df[target_col].nunique() <= 1:
    raise ValueError("CRITICAL: Target variable collapsed to a single value. Model cannot train.")

for col in df.columns:
    if col != target_col and pd.api.types.is_numeric_dtype(df[col]):
        if df[col].std() == 0:
            print(f"WARNING: Feature '{col}' has zero variance after cleaning.")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False, encoding='utf-8')
print("CLEANING_SUCCESS")