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
