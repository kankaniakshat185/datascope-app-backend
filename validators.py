import pandas as pd
import numpy as np

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
            
    return issues
