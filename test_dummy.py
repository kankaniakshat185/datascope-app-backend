import pandas as pd
from impact_engine import _get_metric_baseline, calculate_impact

df = pd.DataFrame({
    "Peak": [None, 1, 2, 3, 4, 5, 2, None, None, None, 1, 2, 3],
    "All Time Peak": [None, None, 1, 2, 3, 4, 5, None, None, 5, 6, 7, 8],
    "Shows": [100, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    "Target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] 
})

print("Baseline:")
print(_get_metric_baseline(df, "Target"))

issue = {"type": "missing_values", "column": "Peak"}
print("\nImpact:")
print(calculate_impact(df, "Target", issue))
