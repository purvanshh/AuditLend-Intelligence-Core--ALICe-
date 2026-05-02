# Phase 9 Heuristic vs ML Benchmark

Run ID: `XGB_V1`
Selected candidate: `XGB_V1`
Threshold: `0.50`

## Assumptions

- Heuristic benchmark uses a deterministic income-stability proxy derived from the engineered feature set.
- ML benchmark uses calibrated probabilities when available.
- Simulated profit uses `+12%` of loan amount for performing approved loans and `-65%` loss given default for approved loans that default.

## Arm Comparison

| Arm | Rows | Approval Rate | Decline Rate | Review Rate | Avg Confidence | Default Rate on Approved | Simulated Profit | Profit / App |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic | 49230 | 0.8514 | 0.0000 | 0.1486 | 0.9007 | 0.1505 | -9354600.50 | -190.02 |
| ml | 49230 | 0.8575 | 0.1425 | 0.0000 | 0.9670 | 0.0235 | 58939506.50 | 1197.23 |

## ML Minus Heuristic

- Approval rate delta: `0.0062`
- Default rate delta on approved loans: `-0.1271`
- Simulated profit delta: `68294107.00`
