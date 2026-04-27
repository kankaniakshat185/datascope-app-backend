import pandas as pd
import numpy as np
import re

def detect_pii(df: pd.DataFrame) -> list:
    pii_issues = []
    
    pii_patterns = {
        "Email Addresses": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        "Phone Numbers": r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "Social Security Numbers": r'\b\d{3}-\d{2}-\d{4}\b',
        "Credit Card Numbers": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
    }
    
    for col in df.select_dtypes(include=['object', 'string']).columns:
        col_data = df[col].dropna().astype(str)
        if len(col_data) == 0:
            continue
            
        sample_size = min(1000, len(col_data))
        sample_data = col_data.sample(sample_size, random_state=42)
        
        for pii_type, pattern in pii_patterns.items():
            matches = sample_data.str.contains(pattern, regex=True).sum()
            match_percentage = (matches / sample_size) * 100
            
            if match_percentage > 5:
                pii_issues.append({
                    "type": "pii_detected",
                    "column": col,
                    "pii_type": pii_type,
                    "percentage": match_percentage
                })
                break
                
    return pii_issues

def run_validators(df: pd.DataFrame) -> list:
    issues = []
    
    # 1. Missing Values Validation
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            percentage = (count / len(df)) * 100
            issues.append({
                "type": "missing_values",
                "column": col,
                "percentage": percentage,
                "count": count
            })
            
    # 2. Uniqueness Validation for ID-like columns
    for col in df.columns:
        if "id" in col.lower() or "uuid" in col.lower() or df[col].is_unique:
            if not df[col].is_unique:
                duplicates = df.duplicated(subset=[col]).sum()
                issues.append({
                    "type": "uniqueness_violation",
                    "column": col,
                    "count": duplicates,
                    "percentage": (duplicates / len(df)) * 100
                })
                
    # 3. Categorical Validation (Cardinality check)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_vals = df[col].nunique()
        if unique_vals > len(df) * 0.9 and unique_vals > 50:
            issues.append({
                "type": "high_cardinality",
                "column": col,
                "count": unique_vals,
            })
            
    # 4. PII Detection
    pii_issues = detect_pii(df)
    issues.extend(pii_issues)
            
    return issues
