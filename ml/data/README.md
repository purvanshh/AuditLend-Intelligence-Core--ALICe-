# Lending Club Data

Place the Lending Club accepted-loans dataset in `ml/data/raw/` and keep it out of git and Docker image layers.

## Canonical default path

The training pipeline resolves the dataset from `LENDING_CLUB_DATA_PATH` and falls back to:

```text
ml/data/raw/accepted_2007_to_2018Q4.csv.gz
```

## Current local note

If your Kaggle export is still unpacked into nested folders, point the environment variable at the actual CSV file until you flatten or recompress it.

Example for the current workspace layout:

```bash
export LENDING_CLUB_DATA_PATH="ml/data/raw/Lending Club Loan Data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
```

## Kaggle download

If you have the Kaggle CLI configured, this is the reproducible download command:

```bash
kaggle datasets download -d wordsforthewise/lending-club
```

After downloading, move or extract the accepted-loans file into `ml/data/raw/` and set `LENDING_CLUB_DATA_PATH` if the filename differs from the canonical default.

## Phase 2 Outputs

Phase 2 adds deterministic ingestion, feature engineering, and time-based split utilities:

- `ml/data/ingestion.py`: filtering, cleaning, and profiling
- `ml/data/features.py`: 25+ engineered features aligned to AuditLend inputs
- `ml/data/splits.py`: train/validation/test partitioning for the 2007-2018Q4 corpus

Generated reports are written to `ml/data/reports/`.

Because the local accepted-loans corpus ends on `2018-12-01`, the current official working split is:

- Train: `2007-01-01` through `2016-12-31`
- Validation: `2017-01-01` through `2017-12-31`
- Test: `2018-01-01` through `2018-12-31`

This matches the PRD-aligned split used by the official `XGB_V1` training and evaluation workflow while staying faithful to the latest quarter currently available in the local corpus.
