import pandas as pd
from validators import run_validators
from ml_checks import run_ml_checks
from impact_engine import calculate_impact
from suggestions import format_suggestions

def run_all_checks(df: pd.DataFrame, target_col: str):
    """
    Main orchestrator for dataset debugger.
    """
    issues = []
    
    # 1. Validators (Missing values, datatypes, uniqueness, structural etc)
    validation_issues = run_validators(df)
    issues.extend(validation_issues)
    
    # 2. ML Checks (Imbalance, Outliers, Correlation)
    ml_issues = run_ml_checks(df, target_col)
    issues.extend(ml_issues)
    
    final_results = []
    
    # 3 & 4. Impact Engine & Suggestions Formatting
    for issue in issues:
        # Calculate impact of fixing the issue
        impact = calculate_impact(df, target_col, issue)
        
        # Format human-readable output and get severity
        formatted_issue = format_suggestions(issue, impact)
        final_results.append(formatted_issue)
        
    # Sort by severity (HIGH > MEDIUM > LOW)
    severity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    final_results.sort(key=lambda x: severity_order.get(x["severity"], 0), reverse=True)
    
    return final_results
