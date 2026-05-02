# Phase 2 Feature Correlation Heatmap

Generated on 2026-05-02 from the Phase 2 engineered feature set.

Legend:

- `++` strong positive correlation (`>= 0.70`)
- `+` moderate positive correlation (`0.30` to `0.69`)
- `.` weak correlation (`-0.29` to `0.29`)
- `-` moderate negative correlation (`-0.30` to `-0.69`)
- `--` strong negative correlation (`<= -0.70`)

| feature | loan_amount | monthly_income | estimated_existing_emi | dti_ratio | loan_amount_to_income | installment_to_income | credit_score_midpoint | credit_history_age_years | revol_util_ratio | bc_util_ratio | delinquency_burden | recent_inquiry_pressure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loan_amount | 1.00 ++ | 0.27 . | 0.44 + | 0.03 . | 0.56 + | 0.48 + | 0.10 . | 0.17 . | 0.12 . | 0.09 . | -0.02 . | -0.02 . |
| monthly_income | 0.27 . | 1.00 ++ | 0.37 + | -0.15 . | -0.21 . | -0.22 . | 0.09 . | 0.12 . | 0.02 . | -0.01 . | 0.00 . | 0.03 . |
| estimated_existing_emi | 0.44 + | 0.37 + | 1.00 ++ | 0.51 + | -0.11 . | -0.13 . | 0.03 . | 0.22 . | 0.19 . | 0.16 . | -0.03 . | 0.04 . |
| dti_ratio | 0.03 . | -0.15 . | 0.51 + | 1.00 ++ | 0.23 . | 0.24 . | -0.10 . | 0.04 . | 0.18 . | 0.21 . | -0.04 . | 0.00 . |
| loan_amount_to_income | 0.56 + | -0.21 . | -0.11 . | 0.23 . | 1.00 ++ | 0.95 ++ | 0.00 . | -0.02 . | 0.08 . | 0.09 . | -0.04 . | -0.08 . |
| installment_to_income | 0.48 + | -0.22 . | -0.13 . | 0.24 . | 0.95 ++ | 1.00 ++ | -0.05 . | -0.05 . | 0.09 . | 0.12 . | -0.02 . | -0.05 . |
| credit_score_midpoint | 0.10 . | 0.09 . | 0.03 . | -0.10 . | 0.00 . | -0.05 . | 1.00 ++ | 0.11 . | -0.46 - | -0.45 - | -0.17 . | -0.09 . |
| credit_history_age_years | 0.17 . | 0.12 . | 0.22 . | 0.04 . | -0.02 . | -0.05 . | 0.11 . | 1.00 ++ | 0.02 . | 0.02 . | 0.04 . | -0.01 . |
| revol_util_ratio | 0.12 . | 0.02 . | 0.19 . | 0.18 . | 0.08 . | 0.09 . | -0.46 - | 0.02 . | 1.00 ++ | 0.76 ++ | -0.01 . | -0.08 . |
| bc_util_ratio | 0.09 . | -0.01 . | 0.16 . | 0.21 . | 0.09 . | 0.12 . | -0.45 - | 0.02 . | 0.76 ++ | 1.00 ++ | -0.01 . | -0.09 . |
| delinquency_burden | -0.02 . | 0.00 . | -0.03 . | -0.04 . | -0.04 . | -0.02 . | -0.17 . | 0.04 . | -0.01 . | -0.01 . | 1.00 ++ | -0.00 . |
| recent_inquiry_pressure | -0.02 . | 0.03 . | 0.04 . | 0.00 . | -0.08 . | -0.05 . | -0.09 . | -0.01 . | -0.08 . | -0.09 . | -0.00 . | 1.00 ++ |
