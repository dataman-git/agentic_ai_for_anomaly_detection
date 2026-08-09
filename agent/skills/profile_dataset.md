## Skill: Profile Dataset

Use this when the user wants to:
- understand the dataset
- inspect schema
- review missing values
- obtain a high-level dataset overview

Action:
- Call `profile_dataset`

Return:
- row count
- column count
- schema
- missingness summary
- numeric summary
- categorical summary

Constraints:
- Do not modify the dataset.
- Do not run anomaly detection automatically.