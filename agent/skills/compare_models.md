## Skill: Compare Models

Use this when the user wants to:
- compare anomaly detection methods
- evaluate different models
- determine which model finds the most anomalies

Actions:
1. Call `compare_models`
2. Explain differences between model outputs

Return:
- model comparison table
- anomaly counts
- anomaly rates
- observations

Constraints:
- Do not automatically select a best model without explanation.