import google.generativeai as genai
import pandas as pd
import os
import csv
import re
import io
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()
from src.utils.pii_scrubber import PIIScrubber

class StewardAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Steward Agent with Gemini 2.5 Flash.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key is required.")
        
        genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash", 
            generation_config={"temperature": 0.2}
        )

    def analyze_data(self, data_path: str, business_objective: str = "") -> Dict[str, str]:
        """
        Analyzes the CSV file and generates a dense textual summary.
        Context-aware: audits based on the business_objective.
        Robustness V3: Implements automatic dialect detection and smart profiling.
        """
        # 1. Detect Encoding
        encodings = ['utf-8', 'latin-1', 'cp1252']
        detected_encoding = 'utf-8' # Default
        
        for enc in encodings:
            try:
                with open(data_path, 'r', encoding=enc) as f:
                    f.read(4096)
                detected_encoding = enc
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        # 2. Detect Dialect (Robust V3)
        dialect_info = self._detect_csv_dialect(data_path, detected_encoding)
        sep = dialect_info['sep']
        decimal = dialect_info['decimal']
        print(f"Steward Detected: Sep='{sep}', Decimal='{decimal}', Encoding='{detected_encoding}'")

        try:
            # 3. Load Data with Fallbacks & Sampling
            file_size = os.path.getsize(data_path)
            file_size_mb = file_size / (1024 * 1024)
            SAMPLE_SIZE = 5000
            
            # Primary Load Attempt
            try:
                if file_size_mb > 10:
                    print(f"Steward: Sampling {SAMPLE_SIZE} rows (File size: {file_size_mb:.2f}MB)")
                    df = pd.read_csv(data_path, sep=sep, decimal=decimal, encoding=detected_encoding, nrows=SAMPLE_SIZE)
                    was_sampled = True
                else:
                    df = pd.read_csv(data_path, sep=sep, decimal=decimal, encoding=detected_encoding)
                    was_sampled = False
            except Exception as e:
                print(f"Steward: Primary load failed ({e}). Attempting fallback engine...")
                # Fallback: Python engine is slower but more robust
                df = pd.read_csv(data_path, sep=sep if sep else None, decimal=decimal, 
                               encoding=detected_encoding, engine='python', on_bad_lines='skip')
                was_sampled = False # Can't guarantee sampling in fallback generic mode easily without nrows, but usually fine
            
            # 4. Standardize & Scrub
            df.columns = df.columns.str.strip().str.replace(' ', '_')
            scrubber = PIIScrubber()
            df = scrubber.scrub_dataframe(df)

            # 5. Smart Profiling (V3)
            profile = self._smart_profile(df, business_objective)
            
            # 6. Construct Prompt
            shape = df.shape
            metadata_str = f"""
            Rows: {shape[0]} (Estimated/Sampled: {was_sampled}), Columns: {shape[1]}
            Filesize: {file_size_mb:.2f} MB
            
            KEY COLUMNS (Top 50 Importance):
            {profile['column_details']}
            
            {profile['alerts']}
            
            Potential IDs: {profile['ids']}
            Potential Dates: {profile['dates']}
            Target Candidates: {profile['targets']}
            
            Example Rows (Random Sample):
            {profile['examples']}
            """
            
            from src.utils.prompting import render_prompt
            
            SYSTEM_PROMPT_TEMPLATE = """
            You are the Senior Data Steward.
            
            MISSION: Support the Business Objective: "$business_objective"
            
            INPUT DATA PROFILE:
            $metadata_str
            
            INSTRUCTIONS:
            1. Start strictly with "DATA SUMMARY:".
            2. Infer the Business Domain (e.g., Retail, CRM, Manufacturing) based on column names.
            3. Explain the *meaning* of key variables relative to the Objective: "$business_objective".
            4. Highlight Data Quality Blockers (e.g., "Target variable 'churn' has 90% nulls").
            5. Mention which columns seem to be Identifiers vs Dates vs Numerical Features.
            6. IF "Sampled: True" is in the profile, YOU MUST EXPLICITLY STATE: "Note: Analysis based on a sample of the first 5000 rows."
            7. Be concise. NO markdown tables. Plain text only.
            """
            
            system_prompt = render_prompt(
                SYSTEM_PROMPT_TEMPLATE,
                business_objective=business_objective,
                metadata_str=metadata_str
            )

            response = self.model.generate_content(system_prompt)
            summary = (getattr(response, "text", "") or "").strip()

            # Diagnostic logging for empty responses (best-effort, no PII)
            try:
                text_len = len(getattr(response, "text", "") or "")
                candidates = getattr(response, "candidates", None)
                cand_count = len(candidates) if candidates is not None else 0
                first = candidates[0] if cand_count else None
                finish_reason = getattr(first, "finish_reason", None) if first else None
                safety_ratings = getattr(first, "safety_ratings", None) if first else None
                citation_metadata = getattr(first, "citation_metadata", None) if first else None
                citations = getattr(first, "citations", None) if first else None
                prompt_feedback = getattr(response, "prompt_feedback", None)
                print(f"STEWARD_LLM_DIAG: text_len={text_len} candidates={cand_count} finish_reason={finish_reason}")
                error_classification = None
                if text_len == 0:
                    pf_safety = getattr(prompt_feedback, "safety_ratings", None) if prompt_feedback else None
                    pf_block = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
                    citation_info = citation_metadata or citations
                    print(
                        f"STEWARD_LLM_EMPTY_RESPONSE: finish_reason={finish_reason} safety={safety_ratings} "
                        f"prompt_feedback={{'block_reason': {pf_block}, 'safety': {pf_safety}}} citations={citation_info} "
                        f"prompt_length_chars={len(system_prompt)}"
                    )
                    error_classification = "EMPTY"
                elif text_len < 50:
                    error_classification = "TOO_SHORT"
                trace = {
                    "model": self.model.model_name,
                    "response_text_len": text_len,
                    "prompt_text_len": len(system_prompt),
                    "candidates": cand_count,
                    "finish_reason": str(finish_reason),
                    "safety_ratings": str(safety_ratings),
                    "timestamp": datetime.utcnow().isoformat(),
                    "error_classification": error_classification,
                }
                try:
                    os.makedirs("data", exist_ok=True)
                    import json as _json
                    with open("data/steward_llm_trace.json", "w", encoding="utf-8") as f:
                        _json.dump(trace, f, indent=2)
                except Exception:
                    pass
            except Exception as diag_err:
                print(f"STEWARD_LLM_DIAG_WARNING: {diag_err}")
            if not summary or len(summary) < 10:
                # Fallback deterministic summary to avoid blank output
                shape = df.shape
                cols = [str(c) for c in df.columns[:20]]
                null_sample = df.isna().mean().round(3).to_dict()
                summary = (
                    f"DATA SUMMARY: Fallback deterministic summary. Rows={shape[0]}, Cols={shape[1]}, "
                    f"Columns={cols}. Null_frac_sample={null_sample}"
                )
            
            # Enforce Prefix
            if not summary.startswith("DATA SUMMARY:"):
                summary = "DATA SUMMARY:\n" + summary

            return {
                "summary": summary, 
                "encoding": detected_encoding,
                "sep": sep,
                "decimal": decimal,
                "file_size_bytes": file_size
            }
            
        except Exception as e:
            return {
                "summary": f"DATA SUMMARY: Critical Error analyzing data: {e}", 
                "encoding": detected_encoding,
                "sep": sep, 
                "decimal": decimal
            }

    def _detect_csv_dialect(self, data_path: str, encoding: str) -> Dict[str, str]:
        """
        Robustly detects separator and decimal using csv.Sniffer and internal heuristics.
        """
        try:
            with open(data_path, 'r', encoding=encoding) as f:
                sample = f.read(50000) # 50KB sample
            
            # 1. Delimiter Detection
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=[',', ';', '\t', '|'])
                sep = dialect.delimiter
            except:
                # Fallback Heuristic
                if sample.count(';') > sample.count(','):
                    sep = ';'
                else:
                    sep = ','
            
            # 2. Decimal Detection
            decimal = self._detect_decimal(sample)
            
            return {"sep": sep, "decimal": decimal}
            
        except Exception as e:
            print(f"Steward: Dialect detection failed ({e}). Defaulting to standard.")
            return {"sep": ",", "decimal": "."}

    def _detect_decimal(self, text: str) -> str:
        """
        Analyzes numeric patterns to decide between '.' and ',' as decimal separator.
        """
        # Look for explicit float patterns: 123.45 vs 123,45
        dot_floats = re.findall(r'\d+\.\d+', text)
        comma_floats = re.findall(r'\d+,\d+', text)
        
        # We need to distinguish "comma as thousands sep" from "comma as decimal"
        # Heuristic: If we see many "123,45" but few "123.45", it's likely European.
        # However, "1,000" (thousands) vs "1,000" (small decimal) is hard.
        # Better simple check: 
        # If sep is ';', likely decimal is ','
        # If sep is ',', likely decimal is '.'
        
        # Let's count occurrences
        if len(comma_floats) > len(dot_floats) * 2:
            return ','
        
        return '.'

    def _smart_profile(self, df: pd.DataFrame, objective: str) -> Dict[str, str]:
        """
        Generates intelligent profile: High Card checks, Constant check, Target Detection.
        """
        alerts = ""
        col_details = ""
        ids = []
        dates = []
        targets = []
        
        # Keyword alignment
        obj_tokens = set(re.sub(r'[^a-z0-9]', ' ', objective.lower()).split())
        target_keywords = {'target', 'label', 'churn', 'class', 'outcome', 'y', 'status', 'revenue', 'sales'}
        target_keywords.update(obj_tokens)
        
        # Sort columns by importance (heuristic: keyword match -> numeric -> other)
        # We limit specific details to top 50
        all_cols = df.columns.tolist()
        priority_cols = []
        other_cols = []
        
        for col in all_cols:
            if any(k in col.lower() for k in target_keywords):
                priority_cols.append(col)
                if 'id' not in col.lower():
                    targets.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                priority_cols.append(col)
            else:
                other_cols.append(col)
                
        sorted_cols = (priority_cols + other_cols)[:50]
        
        for col in sorted_cols:
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            from src.utils.missing import is_effectively_missing_series
            null_pct = is_effectively_missing_series(df[col]).mean()
            
            # Cardinality Check
            unique_ratio = n_unique / len(df) if len(df) > 0 else 0
            
            card_tag = ""
            if unique_ratio > 0.98 and n_unique > 50:
                card_tag = "[HIGH CARDINALITY/ID]"
                if 'id' not in col.lower():
                    ids.append(col)
            elif n_unique <= 1:
                card_tag = "[CONSTANT/USELESS]"
                alerts += f"- ALERT: '{col}' is constant (Value: {df[col].dropna().unique()}).\n"
            
            # Date Check (Robust)
            if df[col].dtype == 'object':
                 try:
                    # Sample Check for speed
                    sample_series = df[col].dropna().sample(min(len(df), 100), random_state=42)
                    parsed = pd.to_datetime(sample_series, errors='coerce', dayfirst=True)
                    if parsed.notna().mean() > 0.7:
                        dates.append(col)
                        card_tag += " [DATE-LIKE]"
                 except:
                    pass
            
            col_details += f"- {col}: {dtype}, Unique={n_unique} {card_tag}, Nulls={null_pct:.1%}\n"
            
        # Target Validation
        if targets:
            main_target = targets[0] # Best guess
            if df[main_target].nunique() <= 1:
                alerts += f"\n*** CRITICAL: Potential Target '{main_target}' has NO VARIATION. Modeling impossible. ***\n"

        # Representative Examples
        try:
            examples = df.sample(min(len(df), 3), random_state=42).to_string(index=False)
        except:
            examples = df.head(3).to_string(index=False)
            
        return {
            "column_details": col_details,
            "alerts": alerts,
            "ids": ids,
            "dates": dates,
            "targets": targets,
            "examples": examples
        }
