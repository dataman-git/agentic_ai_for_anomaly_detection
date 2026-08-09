## Skill: Run Anomaly Detection

Use this when the user wants to:
- detect anomalies
- identify outliers
- run anomaly detection models

Actions:
1. Call `run_pyod_model`
2. Call `summarize_anomalies`

Return:
- model used
- anomaly count
- anomaly percentage
- summary of findings

Constraints:
- Do not modify the dataset.
- Do not compare models unless requested.