import pandas as pd
import re
import hashlib

class PIIScrubber:
    """
    Detects and scrubs Personally Identifiable Information (PII) from pandas DataFrames.
    Targets: Emails, Phone Numbers, Credit Cards, IBANs.
    """

    def __init__(self):
        # Compiled Regex Patterns for Performance
        self.patterns = {
            'EMAIL': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', re.IGNORECASE),
            'PHONE': re.compile(r'(?:\+\d{1,3})?[-. (]*\d{3}[-. )]*\d{3}[-. ]*\d{4}', re.IGNORECASE),
            # Simple Credit Card (Luhn not checked here for speed, just pattern)
            # Matches sequences of 13-16 digits with potential spaces/dashes
            'CREDIT_CARD': re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
            # Simple IBAN (Structure check only)
            'IBAN': re.compile(r'[a-zA-Z]{2}\d{2}[a-zA-Z0-9]{4,}', re.IGNORECASE)
        }

    def scrub_dataframe(self, df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
        """
        Scans object columns. If > threshold% of values match a PII pattern,
        anonymizes the entire column using hashing to preserve referential integrity (if needed)
        or simple redaction labels.
        
        Using Hashing (SHA256 truncated) to allow distinct counting if necessary, 
        or '[REDACTED_{TYPE}]' labels.
        
        Chosen Strategy: Label + Hash Prefix (e.g., "EMAIL_1a2b3c...") for readability + uniqueness.
        """
        df_clean = df.copy()
        
        # Iterate over object/string columns only
        object_cols = df_clean.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            # Drop nulls for detection
            valid_values = df_clean[col].dropna().astype(str).tolist()
            if not valid_values:
                continue
            
            # Use a sample for speed if large
            sample_size = min(len(valid_values), 200)
            sample = valid_values[:sample_size]
            
            # Check against patterns
            detected_type = None
            for pii_type, pattern in self.patterns.items():
                match_count = sum(1 for val in sample if pattern.search(val))
                if match_count / sample_size > threshold:
                    detected_type = pii_type
                    break # Priority: Email > Phone > etc. (Order in dict matters if overlap)
            
            if detected_type:
                print(f"🔒 PII Scrubbing: Detected {detected_type} in column '{col}'. Scrubbing...")
                # Apply Anonymization
                df_clean[col] = df_clean[col].apply(lambda x: self._anonymize(x, detected_type))
                
        return df_clean

    def _anonymize(self, value, pii_type):
        if pd.isna(value):
            return value
        
        val_str = str(value)
        # Create a short hash
        hash_digest = hashlib.sha256(val_str.encode()).hexdigest()[:8]
        return f"[{pii_type}_{hash_digest}]"
