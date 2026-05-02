# AuditLend Intelligence Core (ALICe) ML Guide

This directory contains the ML implementation for AuditLend Intelligence Core (ALICe): data ingestion, feature engineering, model training, calibration, explainability, governance, and benchmark tooling.

## Layout

| Path | Purpose |
| --- | --- |
| `ml/data/` | Lending Club dataset handling, feature engineering, split logic, and exploratory reports |
| `ml/models/` | Training, evaluation, calibration, manifests, reports, and model artifacts |
| `ml/explain/` | SHAP-based per-prediction explainability |
| `ml/governance/` | Model registry, drift detection, A/B routing, and experiment summaries |
| `ml/benchmark/` | Heuristic-versus-ML comparison scripts and reports |

## Dataset

Set the Lending Club source file with:

```bash
export LENDING_CLUB_DATA_PATH="ml/data/raw/accepted_2007_to_2018Q4.csv.gz"
```

If your local Kaggle export is unpacked into nested folders, point the variable at the real CSV path instead. See `ml/data/README.md` for the reproducible download command and the canonical folder layout.

## Core Commands

Train:

```bash
python -m ml.models.train
```

Evaluate:

```bash
python -m ml.models.evaluate
```

Calibrate:

```bash
python -m ml.models.calibrate
```

Benchmark heuristic versus deployed ML behavior:

```bash
python -m ml.benchmark.heuristic_vs_ml
```

All of the model commands accept the same capped-sampling style arguments used during local smoke runs, for example:

```bash
python -m ml.models.train --max-rows-per-split 50000 --modulo-sampling 8
python -m ml.models.evaluate --manifest-path ml/models/experiments/<run_id>/manifest.json --max-rows-per-split 50000 --modulo-sampling 8
python -m ml.models.calibrate --manifest-path ml/models/experiments/<run_id>/manifest.json --max-rows-per-split 50000 --modulo-sampling 8
python -m ml.benchmark.heuristic_vs_ml --manifest-path ml/models/experiments/<run_id>/manifest.json --max-rows-per-split 50000 --modulo-sampling 8
```

## Runtime Integration

- The heuristic path remains the default.
- ML scoring is activated when `ML_ENABLED=true`, `RULE_SET_VERSION=RULE_SET_V2`, or when A/B routing assigns the request to the `ml` arm.
- ML scoring writes an `ML_SCORING` audit entry with:
  - `model_version`
  - `scoring_strategy`
  - fallback metadata
  - SHAP-derived factor contributions
- If the model is unavailable or model confidence is below `CONFIDENCE_THRESHOLD`, the system falls back to the heuristic scorer and records that fallback in the audit log.

## Local Verification Snapshot

Latest local smoke artifact set:

- Evaluation report: `ml/models/reports/20260502T184238Z-smoke3_evaluation.md`
- Calibration report: `ml/models/reports/20260502T184238Z-smoke3_calibration.md`
- Benchmark report: `ml/benchmark/reports/20260502T184238Z-smoke3_heuristic_vs_ml.md`

These are useful engineering verification artifacts, but they are still capped local runs rather than full production-scale certification results.
