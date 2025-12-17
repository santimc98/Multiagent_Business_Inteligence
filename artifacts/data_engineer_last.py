import pandas as pd
import numpy as np
import json
import re
import os
from collections import Counter

def safe_convert_numeric_currency(series, decimal_hint=',', thousands_hint='.'):
    """
    Robust localized number parser for EU/US formats.
    Returns parsed series and metadata.
    """
    if series.isna().all():
        return series, {"parse_success_rate": 0.0, "digits_ratio": 0.0, "sample_size": 0}
    
    # Sample up to 200 non-null values for efficiency
    non_null = series.dropna()
    sample_size = min(200, len(non_null))
    if sample_size == 0:
        return series, {"parse_success_rate": 0.0, "digits_ratio": 0.0, "sample_size": 0}
    
    sample = non_null.sample(sample_size, random_state=42)
    sample_str = sample.astype(str)
    
    # Calculate digits ratio in sample
    digits_pattern = re.compile(r'\d')
    has_digits = sample_str.str.contains(digits_pattern)
    digits_ratio = has_digits.mean()
    
    # Helper for parsing individual value
    def parse_value(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()
        if not val_str:
            return np.nan
        
        # Remove currency symbols, spaces, and non-breaking spaces
        val_str = re.sub(r'[€$£¥\s\u00A0]', '', val_str)
        
        # Handle percentages
        if '%' in val_str:
            val_str = val_str.replace('%', '')
            is_percent = True
        else:
            is_percent = False
        
        # Determine format based on separators
        has_comma_decimal = ',' in val_str and '.' in val_str
        only_comma = ',' in val_str and '.' not in val_str
        only_dot = '.' in val_str and ',' not in val_str
        
        if has_comma_decimal:
            # Both separators present - assume comma is decimal, dot is thousands
            try:
                val_str = val_str.replace('.', '').replace(',', '.')
                result = float(val_str)
            except:
                try:
                    # Try reverse: dot as decimal, comma as thousands
                    val_str = str(val).replace(',', '').replace('.', ',')
                    val_str = val_str.replace(',', '.')
                    result = float(val_str)
                except:
                    return np.nan
        elif only_comma:
            # Only comma - could be decimal or thousands
            parts = val_str.split(',')
            if len(parts[-1]) <= 2 and len(parts) > 1:
                # Likely decimal (EU format)
                val_str = val_str.replace(',', '.')
            else:
                # Likely thousands separator
                val_str = val_str.replace(',', '')
        elif only_dot:
            # Only dot - could be decimal or thousands
            parts = val_str.split('.')
            if len(parts[-1]) <= 2 and len(parts) > 1:
                # Likely decimal
                pass
            else:
                # Likely thousands separator (US format)
                val_str = val_str.replace('.', '')
        
        try:
            result = float(val_str)
            if is_percent:
                result = result / 100.0
            return result
        except:
            return np.nan
    
    # Parse sample for success rate calculation
    parsed_sample = sample.apply(parse_value)
    non_null_before = sample.notna().sum()
    non_null_after = parsed_sample.notna().sum()
    parse_success_rate = non_null_after / non_null_before if non_null_before > 0 else 0.0
    
    # If successful, parse entire series
    if parse_success_rate >= 0.9:
        parsed_series = series.apply(parse_value)
        return parsed_series, {
            "parse_success_rate": parse_success_rate,
            "digits_ratio": digits_ratio,
            "decimal_hint": decimal_hint,
            "thousands_hint": thousands_hint,
            "sample_size": sample_size,
            "reverted": False,
            "reason": ""
        }
    else:
        # Revert - keep original
        return series, {
            "parse_success_rate": parse_success_rate,
            "digits_ratio": digits_ratio,
            "decimal_hint": decimal_hint,
            "thousands_hint": thousands_hint,
            "sample_size": sample_size,
            "reverted": True,
            "reason": f"parse_success_rate {parse_success_rate:.3f} < 0.9"
        }

def is_effectively_missing(value):
    """Check if value is effectively missing (None, NaN, empty string, whitespace)."""
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip() == ''
    return False

def norm(s):
    """Normalize column name for matching."""
    return re.sub(r'[^0-9a-zA-Z]+', '', str(s).lower())

def clean_data():
    # Initialize manifest
    manifest = {
        "input_dialect": {"sep": ",", "decimal": ",", "encoding": "utf-8"},
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "original_columns": [],
        "column_mapping": {},
        "dropped_columns": [],
        "conversions": {},
        "conversions_meta": {},
        "rows_before": 0,
        "rows_after": 0
    }
    
    # Load data with robust loading
    input_path = 'data/raw.csv'
    try:
        df = pd.read_csv(input_path, sep=',', decimal=',', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_path, sep=',', decimal=',', encoding='latin1')
            manifest["input_dialect"]["encoding"] = "latin1"
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")
    
    manifest["rows_before"] = len(df)
    manifest["original_columns"] = list(df.columns)
    
    # 1. Canonicalize column names (snake_case with deduplication)
    seen = {}
    new_columns = []
    unknown_counter = 1
    
    for col in df.columns:
        if pd.isna(col) or str(col).strip() == '':
            base_name = f'unknown_col_{unknown_counter}'
            unknown_counter += 1
        else:
            # Convert to snake_case
            col_str = str(col).strip()
            # Replace spaces and special chars with underscore
            col_str = re.sub(r'[^\w\s]', '_', col_str)
            col_str = re.sub(r'\s+', '_', col_str)
            col_str = col_str.lower()
            base_name = col_str
        
        # Deduplicate
        if base_name in seen:
            seen[base_name] += 1
            new_name = f'{base_name}__{seen[base_name]}'
        else:
            seen[base_name] = 1
            new_name = base_name
        
        new_columns.append(new_name)
        manifest["column_mapping"][col] = new_name
    
    df.columns = new_columns
    
    # 2. Initial type preservation (store original dtypes for reference)
    original_dtypes = df.dtypes.astype(str).to_dict()
    
    # 3. Type inference and cleaning
    for col in df.columns:
        col_data = df[col]
        non_null = col_data.dropna()
        
        # Skip if all null
        if len(non_null) == 0:
            continue
        
        # Check if column looks like numeric/currency
        sample = non_null.head(200)
        sample_str = sample.astype(str)
        digits_ratio = sample_str.str.contains(r'\d').mean()
        
        if digits_ratio > 0.3:
            # Try numeric conversion
            parsed, meta = safe_convert_numeric_currency(col_data, decimal_hint=',', thousands_hint='.')
            
            if not meta['reverted']:
                df[col] = parsed
                manifest["conversions"][col] = "clean_currency"
                manifest["conversions_meta"][col] = meta
            else:
                # Check if it's a percentage string (e.g., "33,33%")
                if col == 'plazoconsumido' or '%' in str(sample.iloc[0] if len(sample) > 0 else ''):
                    # Special handling for percentage columns
                    try:
                        # Remove % and replace comma with dot
                        parsed_pct = col_data.astype(str).str.replace('%', '').str.replace(',', '.').astype(float) / 100.0
                        success_rate = parsed_pct.notna().sum() / col_data.notna().sum()
                        if success_rate >= 0.9:
                            df[col] = parsed_pct
                            manifest["conversions"][col] = "percentage"
                            manifest["conversions_meta"][col] = {
                                "parse_success_rate": success_rate,
                                "digits_ratio": digits_ratio,
                                "decimal_hint": ".",
                                "thousands_hint": "",
                                "sample_size": len(sample),
                                "reverted": False,
                                "reason": "percentage conversion"
                            }
                    except:
                        pass
        
        # Check for date columns
        elif col_data.dtype == 'object':
            # Try to parse dates
            try:
                parsed_date = pd.to_datetime(col_data, errors='coerce', dayfirst=True)
                date_success_rate = parsed_date.notna().sum() / col_data.notna().sum()
                if date_success_rate > 0.7 and date_success_rate < 1.0:  # Not all dates, but majority
                    df[col] = parsed_date
                    manifest["conversions"][col] = "date"
                    manifest["conversions_meta"][col] = {
                        "parse_success_rate": date_success_rate,
                        "date_format": "inferred",
                        "reverted": False
                    }
            except:
                pass
    
    # 4. Create required derived columns
    # Handle plazo_consumido conversion (special case based on business requirement)
    plazo_col_candidates = [c for c in df.columns if 'plazo' in c.lower() and 'consumido' in c.lower()]
    if plazo_col_candidates:
        plazo_col = plazo_col_candidates[0]
        if df[plazo_col].dtype == 'object':
            # Clean percentage string
            try:
                df['plazo_consumido_clean'] = (
                    df[plazo_col]
                    .astype(str)
                    .str.replace('%', '')
                    .str.replace(',', '.')
                    .astype(float) / 100.0
                )
            except:
                df['plazo_consumido_clean'] = pd.to_numeric(
                    df[plazo_col].astype(str).str.replace('%', '').str.replace(',', '.'),
                    errors='coerce'
                ) / 100.0
        else:
            df['plazo_consumido_clean'] = df[plazo_col]
        
        # Create adjusted version per business rule
        df['plazoconsumido_adj'] = df['plazo_consumido_clean'].apply(
            lambda x: x if 0 <= x <= 1 else (0 if x > 1 else np.nan)
        )
    else:
        df['plazoconsumido_adj'] = np.nan
    
    # 5. Create RefScore based on case definitions
    def assign_refscore(row):
        # Extract components
        riim10 = row.get('riim10') if 'riim10' in row else np.nan
        fec = row.get('fec') if 'fec' in row else np.nan
        plazo_adj = row.get('plazoconsumido_adj') if 'plazoconsumido_adj' in row else np.nan
        impacto = row.get('impacto') if 'impacto' in row else np.nan
        
        # Handle missing values
        if pd.isna(riim10) or pd.isna(fec) or pd.isna(impacto):
            return -1.0
        
        # Determine Riesgo category
        if riim10 < 3:
            riesgo = 'ALTO'
        elif 4 <= riim10 <= 6:
            riesgo = 'MEDIO'
        elif 7 <= riim10 <= 10:
            riesgo = 'BAJO'
        else:
            return -1.0  # DEFAULT case
        
        # Determine FEC_window
        if -7 <= fec <= 7:
            fec_window = 'CRITICA'
        elif -15 < fec < -7:
            fec_window = 'PRE_CERCANO'
        elif 7 < fec <= 30:
            fec_window = 'VENCIDA_RECIENTE'
        else:
            fec_window = 'COLA'
        
        # Determine Plazo category
        if pd.isna(plazo_adj) or plazo_adj >= 0.66:
            plazo_cat = 'ALTO_CONSUMO'
        else:
            plazo_cat = 'BAJO_CONSUMO'
        
        # Determine Impacto category
        impacto_cat = '+Imp' if impacto == 1 else '-Imp'
        
        # Match cases (in order)
        if riesgo == 'ALTO':
            if fec_window == 'CRITICA':
                if plazo_cat == 'ALTO_CONSUMO' and impacto_cat == '+Imp':
                    return 1.00
                elif plazo_cat == 'BAJO_CONSUMO' and impacto_cat == '+Imp':
                    return 0.90
                elif impacto_cat == '-Imp':
                    return 0.70
            elif fec_window == 'VENCIDA_RECIENTE':
                if plazo_cat == 'ALTO_CONSUMO' and impacto_cat == '+Imp':
                    return 0.85
                elif impacto_cat == '-Imp':
                    return 0.75
            elif fec_window == 'PRE_CERCANO':
                if plazo_cat == 'ALTO_CONSUMO' and impacto_cat == '+Imp':
                    return 0.70
                elif impacto_cat == '-Imp':
                    return 0.50
            elif fec_window == 'COLA':
                return 0.40
        elif riesgo == 'MEDIO':
            if fec_window == 'CRITICA':
                if impacto_cat == '+Imp':
                    return 0.70
                elif impacto_cat == '-Imp':
                    return 0.60
            elif fec_window in ['PRE_CERCANO', 'VENCIDA_RECIENTE']:
                if impacto_cat == '+Imp':
                    return 0.55
                elif impacto_cat == '-Imp':
                    return 0.45
            elif fec_window == 'COLA':
                return 0.35
        elif riesgo == 'BAJO':
            if fec_window == 'CRITICA':
                if plazo_cat == 'ALTO_CONSUMO' and impacto_cat == '+Imp':
                    return 0.65
                elif plazo_cat == 'BAJO_CONSUMO' and impacto_cat == '+Imp':
                    return 0.55
                elif impacto_cat == '-Imp':
                    return 0.35
            elif fec_window in ['PRE_CERCANO', 'VENCIDA_RECIENTE']:
                if impacto_cat == '+Imp':
                    return 0.45
                elif impacto_cat == '-Imp':
                    return 0.30
            elif fec_window == 'COLA':
                return 0.20
        
        # DEFAULT case
        return -1.0
    
    df['refscore'] = df.apply(assign_refscore, axis=1)
    
    # 6. Drop garbage columns (100% null or constant)
    for col in df.columns.copy():
        non_missing = df[col].apply(lambda x: not is_effectively_missing(x)).sum()
        if non_missing == 0:
            df.drop(columns=[col], inplace=True)
            manifest["dropped_columns"].append({
                "name": col,
                "reason": "100% null/empty"
            })
        elif df[col].nunique(dropna=True) <= 1:
            df.drop(columns=[col], inplace=True)
            manifest["dropped_columns"].append({
                "name": col,
                "reason": "constant (nunique <= 1)"
            })
    
    manifest["rows_after"] = len(df)
    
    # 7. Validation against contract
    contract_requirements = {
        "EntityId": {"role": "categorical", "null_frac": 0.0},
        "Importe Norm": {"role": "feature", "null_frac": 0.0, "range": None},
        "RIIM10 Norm": {"role": "feature", "null_frac": 0.0, "range": None},
        "%plazoConsumido": {"role": "percentage", "null_frac": 0.05, "range": [0, 100]},
        "Score FEC": {"role": "feature", "null_frac": 0.0, "range": None},
        "Impacto": {"role": "feature", "null_frac": 0.0, "range": [0, 1]},
        "RIIM10": {"role": "risk_score", "null_frac": 0.0, "range": [1, 10]},
        "FEC": {"role": "feature", "null_frac": 0.0, "range": None},
        "Score": {"role": "feature", "null_frac": 0.0, "range": None},
        "PlazoConsumido_adj": {"role": "percentage", "null_frac": 0.0, "range": [0, 1]},
        "RefScore": {"role": "target", "null_frac": 0.0, "range": [-1.0, 1.0]}
    }
    
    print("CLEANING_VALIDATION:")
    validation_results = []
    
    for req_name, req_spec in contract_requirements.items():
        # Find matching column
        matched_col = None
        for col in df.columns:
            if norm(col) == norm(req_name):
                matched_col = col
                break
        
        result = {
            "column": req_name,
            "actual_column": matched_col,
            "dtype": "",
            "null_fraction": None,
            "null_check": "SKIPPED",
            "range_check": "SKIPPED",
            "range_message": "",
            "message": ""
        }
        
        if matched_col is None:
            result["null_check"] = "MISSING"
            result["range_check"] = "SKIPPED"
            result["message"] = f"Required column not found"
        else:
            col_data = df[matched_col]
            result["dtype"] = str(col_data.dtype)
            
            # Null check
            try:
                null_count = col_data.isna().sum()
                total = len(col_data)
                null_frac = null_count / total if total > 0 else 0
                result["null_fraction"] = null_frac
                
                allowed_null = req_spec.get("null_frac")
                if allowed_null is not None:
                    if null_frac <= allowed_null:
                        result["null_check"] = "PASS"
                    else:
                        result["null_check"] = "FAIL"
                        result["message"] = f"null fraction {null_frac:.3f} > allowed {allowed_null}"
                else:
                    result["null_check"] = "NOT_REQUIRED"
            except Exception as e:
                result["null_check"] = "ERROR"
                result["message"] = f"null check failed: {str(e)}"
            
            # Range check
            try:
                expected_range = req_spec.get("range")
                if expected_range is not None and col_data.dtype.kind in 'iufc':
                    non_null = col_data.dropna()
                    if len(non_null) > 0:
                        actual_min = float(non_null.min())
                        actual_max = float(non_null.max())
                        
                        range_min, range_max = expected_range
                        if range_min is not None and actual_min < range_min:
                            result["range_check"] = "FAIL"
                            result["range_message"] = f"min {actual_min:.3f} < allowed {range_min}"
                        elif range_max is not None and actual_max > range_max:
                            result["range_check"] = "FAIL"
                            result["range_message"] = f"max {actual_max:.3f} > allowed {range_max}"
                        else:
                            result["range_check"] = "PASS"
                    else:
                        result["range_check"] = "SKIPPED"
                        result["range_message"] = "no non-null values"
                elif expected_range is not None:
                    result["range_check"] = "SKIPPED"
                    result["range_message"] = f"dtype {col_data.dtype} not numeric"
                else:
                    result["range_check"] = "NOT_REQUIRED"
            except Exception as e:
                result["range_check"] = "ERROR"
                result["range_message"] = f"range check failed: {str(e)}"
        
        validation_results.append(result)
    
    # Print validation table
    print(f"{'Column':<20} {'Actual':<20} {'Null%':<8} {'Null Check':<12} {'Range Check':<12} {'Message':<30}")
    print("-" * 100)
    for res in validation_results:
        nf = res.get('null_fraction')
        null_pct = 'NA' if nf is None else f"{nf:.2%}"
        null_check = res.get('null_check', 'SKIPPED')
        range_check = res.get('range_check', 'SKIPPED')
        message = res.get('range_message') or res.get('message', '')
        
        print(f"{res['column']:<20} {str(res['actual_column']):<20} {null_pct:<8} {null_check:<12} {range_check:<12} {message[:30]:<30}")
    
    # 8. Ensure output directory exists and save files
    os.makedirs('data', exist_ok=True)
    
    # Save cleaned data
    output_path = 'data/cleaned_data.csv'
    df.to_csv(output_path, index=False, sep=',', decimal='.', encoding='utf-8')
    
    # Save manifest
    manifest_path = 'data/cleaning_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    print("\nCLEANING_SUCCESS")
    print(f"Saved cleaned data to: {output_path}")
    print(f"Saved manifest to: {manifest_path}")
    print(f"Rows: {manifest['rows_before']} -> {manifest['rows_after']}")
    print(f"Columns kept: {len(df.columns)}")
    print(f"Columns dropped: {len(manifest['dropped_columns'])}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_data()