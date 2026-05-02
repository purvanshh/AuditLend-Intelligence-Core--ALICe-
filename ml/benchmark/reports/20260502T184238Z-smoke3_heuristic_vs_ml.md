# Phase 9 Heuristic vs ML Benchmark

Run ID: `20260502T184238Z-smoke3`
Selected candidate: `lightgbm`
Confidence threshold: `0.60`

## Assumptions

- Heuristic benchmark uses a deterministic income-stability proxy derived from the engineered feature set.
- ML benchmark uses calibrated probabilities when available and falls back to the heuristic decision when model confidence is below threshold.
- Simulated profit uses `+12%` of loan amount for performing approved loans and `-65%` loss given default for approved loans that default.

## Arm Comparison

| Arm | Rows | Approval Rate | Decline Rate | Review Rate | Avg Confidence | Default Rate on Approved | Simulated Profit | Profit / App |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic | 200 | 0.8050 | 0.0000 | 0.1950 | 0.8830 | 0.1429 | -17195.00 | -85.97 |
| ml | 200 | 0.6800 | 0.1450 | 0.1750 | 0.9095 | 0.0147 | 206466.00 | 1032.33 |

## ML Minus Heuristic

- Approval rate delta: `-0.1250`
- Default rate delta on approved loans: `-0.1282`
- Simulated profit delta: `223661.00`
