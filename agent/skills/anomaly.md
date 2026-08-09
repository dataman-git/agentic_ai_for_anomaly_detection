## Skill: Anomaly Detection and Summary

When asked to do detect anomalies, do the following:

1. Select anomaly detection method:
   - small dataset → z-score
   - large dataset → isolation_forest

2. Call selected detection tool

3. Call `summarize_anomalies`

Return:
- method used
- anomaly count
- anomaly percentage
- key patterns