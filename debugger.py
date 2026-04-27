import pandas as pd
from validators import run_validators
from ml_checks import run_ml_checks
from impact_engine import calculate_impact
from suggestions import format_suggestions

import pandas as pd
from validators import run_validators
from ml_checks import run_ml_checks
from impact_engine import calculate_impact
from suggestions import format_suggestions

def run_all_checks(df: pd.DataFrame, target_col: str, custom_rules: list = None):
    """
    Main orchestrator for dataset debugger.
    """
    issues = []

    # 1. Validators
    validation_issues = run_validators(df)
    issues.extend(validation_issues)

    # 2. ML Checks
    ml_issues = run_ml_checks(df, target_col)
    issues.extend(ml_issues)
    
    # 2.5 Custom Rules
    if custom_rules:
        from validators import run_custom_rules
        custom_issues = run_custom_rules(df, custom_rules)
        issues.extend(custom_issues)

    final_results = []

    # 3 & 4. Impact Engine + Suggestions
    for issue in issues:
        impact_data = calculate_impact(df, target_col, issue)

        impact = impact_data["impact"]
        baseline_score = impact_data["baseline_score"]
        after_score = impact_data["after_score"]
        metric = impact_data["metric"]

        # Format output
        formatted_issue = format_suggestions(issue, impact)

        # ✅ Attach metrics to final output
        formatted_issue["baseline_score"] = baseline_score
        formatted_issue["after_score"] = after_score
        formatted_issue["metric"] = metric
        
        if "confidence_score" in impact_data:
            formatted_issue["confidence_score"] = impact_data["confidence_score"]
            
        if "eval_metrics" in impact_data:
            formatted_issue["eval_metrics"] = impact_data["eval_metrics"]

        final_results.append(formatted_issue)

    # Sort by impact first, then severity
    severity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

    final_results.sort(
        key=lambda x: (x["impact"], severity_order.get(x["severity"], 0)),
        reverse=True
    )

    # Total impact (only positive)
    total_impact = sum([
        item["impact"] for item in final_results
        if isinstance(item["impact"], (int, float)) and item["impact"] > 0
    ])

    return {
        "issues": final_results,
        "total_impact": round(total_impact, 2)
    }