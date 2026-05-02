# Phase 3 Training Pipeline

Run the training pipeline with:

```bash
python -m ml.models.train
```

Useful local options:

```bash
python -m ml.models.train --max-rows-per-split 50000 --modulo-sampling 8
```

Outputs:

- `ml/models/experiments/<run_id>/search_results.jsonl`
- `ml/models/experiments/<run_id>/manifest.json`
- `ml/models/artifacts/<run_id>/<best_model>.pkl`

Default model families:

- Logistic Regression baseline
- XGBoost classifier
- LightGBM classifier

Model selection uses validation AUC-PR by default.

## Phase 4 Evaluation

Evaluate the latest manifest with:

```bash
python -m ml.models.evaluate
```

Or evaluate a specific run with the same capped sampling used during training:

```bash
python -m ml.models.evaluate \
  --manifest-path ml/models/experiments/<run_id>/manifest.json \
  --max-rows-per-split 50000 \
  --modulo-sampling 8
```

Evaluation outputs:

- `ml/models/reports/<run_id>_evaluation.md`
- split metrics with AUC-ROC, AUC-PR, Brier score, and ECE
- threshold tables for confusion-matrix analysis
- model-family comparison from the Phase 3 search log

## Phase 5 Calibration

Fit isotonic regression on the validation split and persist the calibrator with:

```bash
python -m ml.models.calibrate
```

Or calibrate a specific run:

```bash
python -m ml.models.calibrate \
  --manifest-path ml/models/experiments/<run_id>/manifest.json \
  --max-rows-per-split 50000 \
  --modulo-sampling 8
```

Calibration outputs:

- `ml/models/artifacts/<run_id>/isotonic_calibrator.pkl`
- `ml/models/artifacts/<run_id>/isotonic_calibration_manifest.json`
- `ml/models/reports/<run_id>_calibration.md`
- `ml/models/reports/<run_id>_validation_calibration.svg`
- `ml/models/reports/<run_id>_test_calibration.svg`
