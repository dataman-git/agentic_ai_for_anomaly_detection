## Skill: Explain Anomalies

Use this when the user wants to:
- understand why observations were flagged
- interpret anomaly detection results
- investigate anomaly drivers

Actions:
1. Call `get_top_anomalies`
2. Call `explain_anomalies`

Return:
- top anomalies
- anomaly scores
- influential features
- explanation of anomaly drivers

Constraints:
- Require anomaly detection results to exist first.
- Do not rerun models unless requested.