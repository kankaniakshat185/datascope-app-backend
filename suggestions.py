def format_suggestions(issue: dict, impact_percentage: float) -> dict:
    """
    Transforms raw issue dictionary into UI-ready suggestion with severity and impact.
    """
    issue_type = issue.get("type")
    
    # Assign default severity based on impact
    if impact_percentage > 5.0 or issue_type in ["data_leakage", "class_imbalance"]:
        severity = "HIGH"
    elif impact_percentage > 2.0 or issue_type in ["high_correlation", "missing_values", "outliers"]:
        severity = "MEDIUM"
    else:
        severity = "LOW"
        
    # Hardcode data leakage to high
    if issue_type == "data_leakage": severity = "HIGH"
        
    formatted = {
        "type": issue_type,
        "severity": severity,
        "impact": f"+{impact_percentage}%" if impact_percentage >= 0 else f"{impact_percentage}%",
        "description": "",
        "suggestion": ""
    }
    
    if issue_type == "missing_values":
        col = issue.get("column")
        perc = issue.get("percentage")
        formatted["description"] = f"Missing values in '{col}' ({perc:.1f}%)"
        formatted["suggestion"] = "Median Imputation" if "age" in col.lower() or "price" in col.lower() else "Impute missing values"
        
    elif issue_type == "class_imbalance":
        ratio = issue.get("ratio")
        formatted["description"] = f"Class Imbalance ({ratio})"
        formatted["suggestion"] = "Apply SMOTE or Class Weights"
        
    elif issue_type == "high_correlation":
        col = issue.get("column")
        corrs = ", ".join(issue.get("correlated_with", []))
        formatted["description"] = f"Highly correlated features ({col}, {corrs})"
        formatted["suggestion"] = f"Drop redundant column: {col}"
        
    elif issue_type == "outliers":
        perc = issue.get("percentage")
        formatted["description"] = f"Outliers detected ({perc:.1f}% of data)"
        formatted["suggestion"] = "Remove or cap extreme values (e.g. IQR method)"
        
    elif issue_type == "data_leakage":
        col = issue.get("column")
        formatted["description"] = f"Data Leakage: '{col}' is perfectly predicting the target"
        formatted["suggestion"] = f"Drop '{col}' to prevent model memorization"
        
    elif issue_type == "uniqueness_violation":
        col = issue.get("column")
        formatted["description"] = f"ID column '{col}' has duplicate rows"
        formatted["suggestion"] = "Remove duplicate IDs"
        
    elif issue_type == "high_cardinality":
        col = issue.get("column")
        formatted["description"] = f"Categorical column '{col}' has too many unique values"
        formatted["suggestion"] = "Drop column or apply target encoding"
        
    else:
        formatted["description"] = f"Generic issue detected: {issue_type}"
        formatted["suggestion"] = "Investigate further"
        
    return formatted
