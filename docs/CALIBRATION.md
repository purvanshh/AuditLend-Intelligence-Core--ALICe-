# Score Calibration

**Current Status:** Dual-path scoring with a default heuristic scorecard and an opt-in ML-assisted scorer.

## RULE_SET_V1 (2025-01-15)

- **Methodology:** SME-derived weights with conservative thresholds.
- **Validation:** Not yet empirically validated against historical default data.
- **Known limitation:** This is a governed scorecard, not a statistically calibrated probability-of-default model.
- **Next steps:** Backtest against historical loan performance data; tune using KS, Gini/AUC, calibration curves, and adverse-action review.

## RULE_SET_V2 (2026-05-03)

- **Methodology:** Calibrated ML probability-of-default mapped to a 0-100 risk score, with the heuristic scorer retained as the fallback path.
- **Activation:** Used only when ML is enabled explicitly or when A/B routing assigns the request to the ML arm.
- **Guardrails:** If model confidence is below `CONFIDENCE_THRESHOLD`, if artifacts are unavailable, or if `failure_flags.ml_model` forces a timeout/low-confidence scenario, the system falls back to the heuristic scorer and audits that fallback.
- **Auditability:** `ML_SCORING` entries include `model_version`, `scoring_strategy`, fallback metadata, and SHAP-style feature contributions.

## ML Evaluation Scaffold (2026-05-03)

- The repository now includes a held-out evaluation workflow in `ml/models/evaluate.py`.
- Evaluation reports include AUC-ROC, AUC-PR, Brier score, expected calibration error, threshold tables, and candidate-family comparison.
- The official `XGB_V1` evaluation run on the PRD-aligned full split reported:
  - Raw test AUC-ROC: `0.975786`
  - Calibrated test AUC-ROC: `0.975664`
  - Raw test Brier score: `0.026582`
  - Calibrated test Brier score: `0.025293`
  - Raw test ECE: `0.016177`
  - Calibrated test ECE: `0.003550`

## ML Calibration Scaffold (2026-05-03)

- The repository now includes isotonic calibration in `ml/models/calibrate.py`.
- Calibration is fit on the validation split and then re-evaluated on the held-out test split before any downstream use.
- Each calibration run persists the calibrator artifact and emits before/after reliability reports plus SVG calibration curves.
- The official `XGB_V1` calibrator materially improved held-out test calibration:
  - Raw test ECE: `0.016177`
  - Calibrated test ECE: `0.003550`
  - Raw test Brier: `0.026582`
  - Calibrated test Brier: `0.025293`

## ML Benchmark Snapshot (2026-05-03)

- The benchmark script `ml/benchmark/heuristic_vs_ml.py` compares the heuristic control against the deployed ML strategy on the held-out test split.
- The official `XGB_V1` benchmark at threshold `0.50` reported:
  - Heuristic approval rate: `0.851371`
  - ML approval rate: `0.857526`
  - Heuristic default rate on approved loans: `0.150550`
  - ML default rate on approved loans: `0.023498`
  - Simulated profit delta, ML minus heuristic: `68294107.00`
- On this held-out benchmark, `XGB_V1` achieved both a modestly higher approval rate and a materially lower approved-loan default rate.

## Adding A New Rule Set

1. Create a new immutable `RuleSet` instance in `engine/rule_sets.py` with a unique version.
2. Add it to `ALL_RULE_SETS`.
3. Update `ACTIVE_RULE_SET`.
4. Update `tests/unit/test_rule_governance.py` so the expected active version changes deliberately.
5. Document methodology, validation data, approval date, and known limitations here.
