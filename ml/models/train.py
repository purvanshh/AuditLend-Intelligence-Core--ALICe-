"""Training pipeline for AuditLend Phase 3 model experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import product
from math import exp
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

from ml.data.features import FEATURE_COLUMNS, build_feature_row
from ml.data.ingestion import DEFAULT_STATUS_MAP, MODELING_COLUMNS, ensure_lending_club_data_path, iter_clean_lending_club_rows
from ml.data.splits import assign_time_split

MODEL_NUMERIC_FEATURES: tuple[str, ...] = tuple(column for column in FEATURE_COLUMNS if column != "target_defaulted")
MODEL_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "grade",
    "sub_grade",
    "purpose",
    "home_ownership",
    "verification_status",
)
DEFAULT_SEARCH_METRIC = "validation_auc_pr"
OFFICIAL_MODEL_VERSION = "XGB_V1"
OFFICIAL_MODEL_ARTIFACT_PATH = Path("ml/models/XGB_V1_model.pkl")
OFFICIAL_CALIBRATOR_ARTIFACT_PATH = Path("ml/models/XGB_V1_calibrator.pkl")
OFFICIAL_FEATURE_SPEC_PATH = Path("ml/models/XGB_V1_features.json")
OFFICIAL_MANIFEST_PATH = Path("ml/models/manifest.yaml")
OFFICIAL_SEARCH_RESULTS_PATH = Path("ml/models/XGB_V1_search_results.jsonl")
OFFICIAL_INPUT_FEATURES: tuple[str, ...] = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
OFFICIAL_PROXY_FEATURES: tuple[str, ...] = ("zip_code_prefix", "employment_length_band")


@dataclass(frozen=True)
class MatrixSplit:
    """Encoded feature matrix and target vector for one dataset split."""

    name: str
    feature_names: list[str]
    rows: list[dict[str, Any]]
    X: list[list[float]]
    y: list[int]

    @property
    def row_count(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class PreparedDataset:
    """Encoded train/validation/test matrices plus schema metadata."""

    split_matrices: dict[str, MatrixSplit]
    feature_names: list[str]
    categories_by_feature: dict[str, list[str]]
    data_hash: str


@dataclass(frozen=True)
class MetricSet:
    """Binary classification metrics used during model selection."""

    auc_roc: float
    auc_pr: float
    brier_score: float
    positive_rate: float
    row_count: int


@dataclass(frozen=True)
class CandidateDefinition:
    """Model family definition and search grid."""

    name: str
    family: str
    factory: Callable[[dict[str, Any]], Any]
    parameter_grid: list[dict[str, Any]]


@dataclass(frozen=True)
class CandidateRun:
    """One evaluated candidate configuration."""

    candidate_name: str
    family: str
    params: dict[str, Any]
    training_metrics: MetricSet
    validation_metrics: MetricSet
    test_metrics: MetricSet
    feature_importance: list[dict[str, float]]
    search_metric: float


@dataclass(frozen=True)
class TrainingConfig:
    """Runtime configuration for Phase 3 training."""

    seed: int = 42
    search_metric: str = DEFAULT_SEARCH_METRIC
    max_rows_per_split: int | None = None
    modulo_sampling: int = 1
    experiment_dir: Path = Path("ml/models/experiments")
    artifact_dir: Path = Path("ml/models/artifacts")
    run_label: str = "phase3"
    include_xgboost: bool = True
    include_lightgbm: bool = True
    include_logistic_regression: bool = True


@dataclass(frozen=True)
class TrainingRunSummary:
    """Serializable summary of a full training run."""

    run_id: str
    created_at: str
    data_path: str
    data_hash: str
    split_counts: dict[str, int]
    feature_count: int
    feature_names: list[str]
    categories_by_feature: dict[str, list[str]]
    search_metric: str
    available_candidates: list[str]
    selected_candidate: str
    selected_params: dict[str, Any]
    metrics: dict[str, dict[str, float]]
    feature_importance: list[dict[str, float]]
    artifact_path: str
    experiment_log_path: str
    manifest_path: str


@dataclass(frozen=True)
class OfficialPreparedDataset:
    """Vectorized train/validation/test frames for the official XGB_V1 training path."""

    split_frames: dict[str, Any]
    feature_columns: list[str]
    split_counts: dict[str, int]
    data_hash: str


@dataclass(frozen=True)
class OfficialTrainingSummary:
    """Serializable summary for the signed-off XGB_V1 artifact set."""

    model_version: str
    created_at: str
    data_path: str
    data_hash: str
    split_counts: dict[str, int]
    selected_params: dict[str, Any]
    validation_metrics_raw: dict[str, float | int]
    validation_metrics_calibrated: dict[str, float | int]
    test_metrics_raw: dict[str, float | int]
    test_metrics_calibrated: dict[str, float | int]
    top_feature_importance: list[dict[str, float]]
    model_artifact_path: str
    calibrator_artifact_path: str
    feature_spec_path: str
    manifest_path: str
    search_results_path: str


def prepare_training_dataset(
    *,
    max_rows_per_split: int | None = None,
    modulo_sampling: int = 1,
    env_var: str = "LENDING_CLUB_DATA_PATH",
) -> PreparedDataset:
    """Load, split, encode, and hash the Phase 2 feature set."""

    split_feature_rows = _build_split_feature_rows(
        max_rows_per_split=max_rows_per_split,
        modulo_sampling=modulo_sampling,
        env_var=env_var,
    )
    categories_by_feature = infer_categories(split_feature_rows.get("train", []))
    encoded_splits = encode_split_feature_rows(split_feature_rows, categories_by_feature)
    feature_names = encoded_splits["train"].feature_names if encoded_splits["train"].feature_names else []
    data_hash = compute_data_hash(split_feature_rows)
    return PreparedDataset(
        split_matrices=encoded_splits,
        feature_names=feature_names,
        categories_by_feature=categories_by_feature,
        data_hash=data_hash,
    )


def infer_categories(feature_rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Build deterministic categorical vocabularies from the training split."""

    categories_by_feature: dict[str, list[str]] = {}
    for feature_name in MODEL_CATEGORICAL_FEATURES:
        values = {normalize_category_value(row.get(feature_name)) for row in feature_rows}
        values.add("UNKNOWN")
        categories_by_feature[feature_name] = sorted(values)
    return categories_by_feature


def encode_split_feature_rows(
    split_feature_rows: dict[str, list[dict[str, Any]]],
    categories_by_feature: dict[str, list[str]],
) -> dict[str, MatrixSplit]:
    """One-hot encode categoricals and build matrices for each split."""

    feature_names = build_model_feature_names(categories_by_feature)
    split_matrices: dict[str, MatrixSplit] = {}
    for split_name, rows in split_feature_rows.items():
        X: list[list[float]] = []
        y: list[int] = []
        for row in rows:
            X.append(encode_feature_row(row, categories_by_feature))
            y.append(int(row.get("target_defaulted", 0)))
        split_matrices[split_name] = MatrixSplit(
            name=split_name,
            feature_names=feature_names,
            rows=rows,
            X=X,
            y=y,
        )
    return split_matrices


def build_model_feature_names(categories_by_feature: dict[str, list[str]]) -> list[str]:
    """Return the stable training feature order."""

    feature_names = list(MODEL_NUMERIC_FEATURES)
    for categorical_feature in MODEL_CATEGORICAL_FEATURES:
        for category in categories_by_feature.get(categorical_feature, []):
            feature_names.append(f"{categorical_feature}={category}")
    return feature_names


def encode_feature_row(
    feature_row: dict[str, Any],
    categories_by_feature: dict[str, list[str]],
) -> list[float]:
    """Encode one feature row into the training feature vector order."""

    encoded: list[float] = [_safe_float(feature_row.get(feature_name)) for feature_name in MODEL_NUMERIC_FEATURES]
    for categorical_feature in MODEL_CATEGORICAL_FEATURES:
        observed_value = normalize_category_value(feature_row.get(categorical_feature))
        categories = categories_by_feature.get(categorical_feature, [])
        for category in categories:
            encoded.append(1.0 if observed_value == category else 0.0)
    return encoded


def compute_data_hash(split_feature_rows: dict[str, list[dict[str, Any]]]) -> str:
    """Hash split metadata and row identities for reproducibility tracking."""

    digest = sha256()
    for split_name in ("train", "validation", "test", "holdout"):
        digest.update(split_name.encode("utf-8"))
        for row in split_feature_rows.get(split_name, []):
            digest.update(str(row.get("loan_id", "")).encode("utf-8"))
            digest.update(str(row.get("target_defaulted", 0)).encode("utf-8"))
            digest.update(str(row.get("issue_date", "")).encode("utf-8"))
    return digest.hexdigest()


def compute_binary_metrics(y_true: Sequence[int], probabilities: Sequence[float]) -> MetricSet:
    """Compute AUC-ROC, AUC-PR, Brier score, and base rate without sklearn."""

    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length.")
    if not y_true:
        raise ValueError("Cannot compute metrics on an empty split.")

    auc_roc = _compute_auc_roc(y_true, probabilities)
    auc_pr = _compute_average_precision(y_true, probabilities)
    brier_score = mean((float(probability) - int(target)) ** 2 for target, probability in zip(y_true, probabilities))
    positive_rate = sum(int(target) for target in y_true) / len(y_true)
    return MetricSet(
        auc_roc=round(auc_roc, 6),
        auc_pr=round(auc_pr, 6),
        brier_score=round(brier_score, 6),
        positive_rate=round(positive_rate, 6),
        row_count=len(y_true),
    )


def build_default_candidates(config: TrainingConfig) -> list[CandidateDefinition]:
    """Create the default model families for Phase 3."""

    candidates: list[CandidateDefinition] = []

    if config.include_logistic_regression:
        logistic_factory = _build_logistic_regression_factory(config.seed)
        if logistic_factory is not None:
            candidates.append(
                CandidateDefinition(
                    name="logistic_regression",
                    family="sklearn",
                    factory=logistic_factory,
                    parameter_grid=_expand_grid(
                        {
                            "C": [0.25, 1.0, 4.0],
                            "solver": ["lbfgs"],
                            "max_iter": [500],
                            "class_weight": ["balanced"],
                        }
                    ),
                )
            )

    if config.include_xgboost:
        xgboost_factory = _build_xgboost_factory(config.seed)
        if xgboost_factory is not None:
            candidates.append(
                CandidateDefinition(
                    name="xgboost",
                    family="xgboost",
                    factory=xgboost_factory,
                    parameter_grid=_expand_grid(
                        {
                            "n_estimators": [200, 350],
                            "max_depth": [4, 6],
                            "learning_rate": [0.05, 0.1],
                            "subsample": [0.8],
                            "colsample_bytree": [0.8],
                            "min_child_weight": [1, 5],
                        }
                    ),
                )
            )

    if config.include_lightgbm:
        lightgbm_factory = _build_lightgbm_factory(config.seed)
        if lightgbm_factory is not None:
            candidates.append(
                CandidateDefinition(
                    name="lightgbm",
                    family="lightgbm",
                    factory=lightgbm_factory,
                    parameter_grid=_expand_grid(
                        {
                            "n_estimators": [200, 350],
                            "num_leaves": [31, 63],
                            "learning_rate": [0.05, 0.1],
                            "subsample": [0.8],
                            "colsample_bytree": [0.8],
                        }
                    ),
                )
            )

    return candidates


def run_training(
    config: TrainingConfig,
    *,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    candidates: list[CandidateDefinition] | None = None,
) -> TrainingRunSummary:
    """Execute the Phase 3 search loop and persist artifacts."""

    data_path = ensure_lending_club_data_path(env_var=env_var)
    prepared_dataset = prepare_training_dataset(
        max_rows_per_split=config.max_rows_per_split,
        modulo_sampling=config.modulo_sampling,
        env_var=env_var,
    )

    candidate_definitions = candidates or build_default_candidates(config)
    if not candidate_definitions:
        raise RuntimeError(
            "No trainable model candidates are available. "
            "Install ML dependencies from requirements-ml.txt or provide custom candidates."
        )

    split_matrices = prepared_dataset.split_matrices
    train_matrix = split_matrices["train"]
    validation_matrix = split_matrices["validation"]
    test_matrix = split_matrices["test"]

    if train_matrix.row_count == 0 or validation_matrix.row_count == 0 or test_matrix.row_count == 0:
        raise RuntimeError("Train, validation, and test splits must all contain rows before training.")

    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{config.run_label}"
    run_artifact_dir = config.artifact_dir / run_id
    run_experiment_dir = config.experiment_dir / run_id
    run_artifact_dir.mkdir(parents=True, exist_ok=True)
    run_experiment_dir.mkdir(parents=True, exist_ok=True)

    candidate_runs: list[CandidateRun] = []
    experiment_rows: list[dict[str, Any]] = []

    for candidate in candidate_definitions:
        for params in candidate.parameter_grid:
            model = candidate.factory(params)
            fit_model(model, train_matrix.X, train_matrix.y, feature_names=train_matrix.feature_names)
            train_probabilities = predict_probabilities(model, train_matrix.X, feature_names=train_matrix.feature_names)
            validation_probabilities = predict_probabilities(
                model,
                validation_matrix.X,
                feature_names=validation_matrix.feature_names,
            )
            test_probabilities = predict_probabilities(model, test_matrix.X, feature_names=test_matrix.feature_names)

            training_metrics = compute_binary_metrics(train_matrix.y, train_probabilities)
            validation_metrics = compute_binary_metrics(validation_matrix.y, validation_probabilities)
            test_metrics = compute_binary_metrics(test_matrix.y, test_probabilities)
            feature_importance = extract_feature_importance(model, train_matrix.feature_names)
            search_metric_value = metric_value(validation_metrics, config.search_metric)

            run = CandidateRun(
                candidate_name=candidate.name,
                family=candidate.family,
                params=params,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                feature_importance=feature_importance,
                search_metric=search_metric_value,
            )
            candidate_runs.append(run)
            experiment_rows.append(candidate_run_to_dict(run, config.search_metric))

    best_run = max(
        candidate_runs,
        key=lambda run: (
            run.search_metric,
            run.validation_metrics.auc_roc,
            -run.validation_metrics.brier_score,
        ),
    )

    best_candidate = next(candidate for candidate in candidate_definitions if candidate.name == best_run.candidate_name)
    best_model = best_candidate.factory(best_run.params)
    combined_X = train_matrix.X + validation_matrix.X
    combined_y = train_matrix.y + validation_matrix.y
    fit_model(best_model, combined_X, combined_y, feature_names=train_matrix.feature_names)

    model_artifact_path = run_artifact_dir / f"{best_run.candidate_name}.pkl"
    with model_artifact_path.open("wb") as handle:
        pickle.dump(best_model, handle)

    experiment_log_path = run_experiment_dir / "search_results.jsonl"
    write_experiment_log(experiment_log_path, experiment_rows)

    manifest_path = run_experiment_dir / "manifest.json"
    summary = TrainingRunSummary(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        data_path=str(data_path),
        data_hash=prepared_dataset.data_hash,
        split_counts={name: matrix.row_count for name, matrix in split_matrices.items()},
        feature_count=len(prepared_dataset.feature_names),
        feature_names=prepared_dataset.feature_names,
        categories_by_feature=prepared_dataset.categories_by_feature,
        search_metric=config.search_metric,
        available_candidates=[candidate.name for candidate in candidate_definitions],
        selected_candidate=best_run.candidate_name,
        selected_params=best_run.params,
        metrics={
            "train": metric_set_to_dict(best_run.training_metrics),
            "validation": metric_set_to_dict(best_run.validation_metrics),
            "test": metric_set_to_dict(best_run.test_metrics),
        },
        feature_importance=best_run.feature_importance[:25],
        artifact_path=str(model_artifact_path),
        experiment_log_path=str(experiment_log_path),
        manifest_path=str(manifest_path),
    )
    write_manifest(manifest_path, summary)
    return summary


def fit_model(
    model: Any,
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    feature_names: Sequence[str] | None = None,
) -> None:
    """Fit an estimator using the expected sklearn-like interface."""

    model.fit(_as_model_input(X, feature_names), list(y))


def predict_probabilities(
    model: Any,
    X: Sequence[Sequence[float]],
    *,
    feature_names: Sequence[str] | None = None,
) -> list[float]:
    """Return positive-class probabilities for a fitted estimator."""

    model_input = _as_model_input(X, feature_names)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)
        return [float(row[1]) for row in probabilities]
    if hasattr(model, "decision_function"):
        scores = [float(score) for score in model.decision_function(model_input)]
        return [_sigmoid(score) for score in scores]
    raise TypeError("Model must implement predict_proba or decision_function.")


def extract_feature_importance(model: Any, feature_names: Sequence[str]) -> list[dict[str, float]]:
    """Extract comparable feature importances across supported model families."""

    if hasattr(model, "named_steps"):
        named_steps = getattr(model, "named_steps")
        if "classifier" in named_steps:
            model = named_steps["classifier"]

    raw_importance: list[float] | None = None

    if hasattr(model, "feature_importances_"):
        raw_importance = [float(value) for value in model.feature_importances_]
    elif hasattr(model, "coef_"):
        coefficients = getattr(model, "coef_")
        if coefficients is not None and len(coefficients):
            raw_importance = [abs(float(value)) for value in coefficients[0]]

    if raw_importance is None:
        return []

    importance_rows = [
        {"feature": feature_name, "importance": round(float(importance), 8)}
        for feature_name, importance in zip(feature_names, raw_importance)
    ]
    return sorted(importance_rows, key=lambda row: row["importance"], reverse=True)


def metric_value(metrics: MetricSet, metric_name: str) -> float:
    """Return the configured selection metric from a MetricSet."""

    mapping = {
        "validation_auc_pr": metrics.auc_pr,
        "validation_auc_roc": metrics.auc_roc,
        "validation_brier_neg": -metrics.brier_score,
    }
    if metric_name not in mapping:
        raise ValueError(f"Unsupported search metric: {metric_name}")
    return mapping[metric_name]


def write_experiment_log(output_path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write file-based experiment tracking rows as JSON lines."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def write_manifest(output_path: str | Path, summary: TrainingRunSummary) -> Path:
    """Persist the final run summary as formatted JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def candidate_run_to_dict(run: CandidateRun, search_metric_name: str) -> dict[str, Any]:
    """Serialize one candidate evaluation row."""

    return {
        "candidate_name": run.candidate_name,
        "family": run.family,
        "params": run.params,
        "search_metric_name": search_metric_name,
        "search_metric_value": round(run.search_metric, 6),
        "training_metrics": metric_set_to_dict(run.training_metrics),
        "validation_metrics": metric_set_to_dict(run.validation_metrics),
        "test_metrics": metric_set_to_dict(run.test_metrics),
        "feature_importance": run.feature_importance[:15],
    }


def metric_set_to_dict(metrics: MetricSet) -> dict[str, float | int]:
    """Serialize metrics into plain JSON-friendly floats."""

    return {
        "auc_roc": metrics.auc_roc,
        "auc_pr": metrics.auc_pr,
        "brier_score": metrics.brier_score,
        "positive_rate": metrics.positive_rate,
        "row_count": metrics.row_count,
    }


class OfficialXGBV1Model:
    """Pickle-friendly preprocessing + XGBoost bundle for the signed-off model."""

    def __init__(self, params: dict[str, Any], *, seed: int = 42) -> None:
        self.params = dict(params)
        self.seed = seed
        self.numeric_features = list(MODEL_NUMERIC_FEATURES)
        self.categorical_features = list(MODEL_CATEGORICAL_FEATURES)
        self.input_features = list(OFFICIAL_INPUT_FEATURES)
        self.preprocessor = None
        self.classifier = None

    def fit(self, X, y):
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from xgboost import XGBClassifier

        frame = X if hasattr(X, "loc") else pd.DataFrame(list(X), columns=self.input_features)
        frame = frame.loc[:, self.input_features].copy()
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", self.numeric_features),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype="float32"),
                    self.categorical_features,
                ),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
        self.classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.seed,
            n_jobs=4,
            tree_method="hist",
            verbosity=0,
            **self.params,
        )
        transformed = self.preprocessor.fit_transform(frame)
        self.classifier.fit(transformed, list(y))
        return self

    def predict_proba(self, X):
        import pandas as pd

        if self.preprocessor is None or self.classifier is None:
            raise RuntimeError("OfficialXGBV1Model must be fit before prediction.")
        frame = X if hasattr(X, "loc") else pd.DataFrame(list(X), columns=self.input_features)
        frame = frame.loc[:, self.input_features].copy()
        transformed = self.preprocessor.transform(frame)
        return self.classifier.predict_proba(transformed)

    @property
    def feature_importances_(self) -> list[float]:
        if self.classifier is None or not hasattr(self.classifier, "feature_importances_"):
            return []
        return [float(value) for value in self.classifier.feature_importances_]

    def get_feature_names(self) -> list[str]:
        if self.preprocessor is None:
            return list(self.input_features)
        categorical = self.preprocessor.named_transformers_.get("categorical")
        if categorical is None:
            return list(self.numeric_features)
        categorical_names = list(categorical.get_feature_names_out(self.categorical_features))
        return list(self.numeric_features) + categorical_names


OfficialXGBV1Model.__module__ = "ml.models.train"


def prepare_official_training_dataset(
    *,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    chunk_size: int = 200_000,
) -> OfficialPreparedDataset:
    """Load the full Lending Club corpus into PRD-aligned split frames for XGB_V1."""

    import pandas as pd

    data_path = ensure_lending_club_data_path(env_var=env_var)
    split_parts: dict[str, list[Any]] = {
        "train": [],
        "validation": [],
        "test": [],
        "holdout": [],
    }
    identity_hash = sha256()

    requested_columns = list(MODELING_COLUMNS) + ["zip_code"]
    available_columns = _available_csv_columns(data_path)
    reader = pd.read_csv(
        data_path,
        usecols=[column for column in requested_columns if column in available_columns],
        chunksize=chunk_size,
        low_memory=False,
    )
    for raw_chunk in reader:
        feature_chunk = _build_official_feature_chunk(raw_chunk)
        if feature_chunk.empty:
            continue
        for split_name, mask in _official_split_masks(feature_chunk).items():
            split_chunk = feature_chunk.loc[mask].copy()
            if split_chunk.empty:
                continue
            split_parts[split_name].append(split_chunk)
            for row in split_chunk.loc[:, ["loan_id", "target_defaulted", "issue_date"]].itertuples(index=False):
                identity_hash.update(str(row[0]).encode("utf-8"))
                identity_hash.update(str(int(row[1])).encode("utf-8"))
                identity_hash.update(str(row[2]).encode("utf-8"))

    feature_columns = list(OFFICIAL_INPUT_FEATURES)
    ordered_columns = ["loan_id", "issue_date"] + list(OFFICIAL_PROXY_FEATURES) + feature_columns + ["target_defaulted"]
    split_frames = {
        split_name: (
            pd.concat(parts, ignore_index=True).loc[:, ordered_columns]
            if parts
            else pd.DataFrame(columns=ordered_columns)
        )
        for split_name, parts in split_parts.items()
    }
    split_counts = {split_name: int(len(frame)) for split_name, frame in split_frames.items()}
    return OfficialPreparedDataset(
        split_frames=split_frames,
        feature_columns=feature_columns,
        split_counts=split_counts,
        data_hash=identity_hash.hexdigest(),
    )


def train_official_xgb_v1(
    *,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    seed: int = 42,
) -> OfficialTrainingSummary:
    """Train, calibrate, and persist the official XGB_V1 artifact set."""

    import pandas as pd
    from sklearn.isotonic import IsotonicRegression

    data_path = ensure_lending_club_data_path(env_var=env_var)
    prepared_dataset = prepare_official_training_dataset(env_var=env_var)
    split_frames = prepared_dataset.split_frames
    train_frame = split_frames["train"]
    validation_frame = split_frames["validation"]
    test_frame = split_frames["test"]
    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise RuntimeError("XGB_V1 training requires non-empty train, validation, and test splits.")

    search_train_frame = _deterministic_search_slice(train_frame, max_rows=200_000)
    search_validation_frame = _deterministic_search_slice(validation_frame, max_rows=60_000)
    parameter_grid = _expand_grid(
        {
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6],
            "subsample": [0.8, 1.0],
        }
    )
    base_params = {
        "n_estimators": 200,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
    }

    train_X = train_frame.loc[:, OFFICIAL_INPUT_FEATURES]
    validation_X = validation_frame.loc[:, OFFICIAL_INPUT_FEATURES]
    test_X = test_frame.loc[:, OFFICIAL_INPUT_FEATURES]
    search_train_X = search_train_frame.loc[:, OFFICIAL_INPUT_FEATURES]
    search_validation_X = search_validation_frame.loc[:, OFFICIAL_INPUT_FEATURES]
    train_y = [int(value) for value in train_frame["target_defaulted"].tolist()]
    validation_y = [int(value) for value in validation_frame["target_defaulted"].tolist()]
    test_y = [int(value) for value in test_frame["target_defaulted"].tolist()]
    search_train_y = [int(value) for value in search_train_frame["target_defaulted"].tolist()]
    search_validation_y = [int(value) for value in search_validation_frame["target_defaulted"].tolist()]

    search_rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_metrics: MetricSet | None = None

    for grid_params in parameter_grid:
        candidate_params = {**base_params, **grid_params}
        candidate_model = OfficialXGBV1Model(candidate_params, seed=seed)
        candidate_model.fit(search_train_X, search_train_y)
        validation_probabilities = predict_probabilities(candidate_model, search_validation_X)
        validation_metrics = compute_binary_metrics(search_validation_y, validation_probabilities)
        test_probabilities = predict_probabilities(candidate_model, test_X)
        test_metrics = compute_binary_metrics(test_y, test_probabilities)
        search_row = {
            "candidate_name": OFFICIAL_MODEL_VERSION,
            "family": "xgboost",
            "params": candidate_params,
            "search_train_rows": len(search_train_frame),
            "search_validation_rows": len(search_validation_frame),
            "validation_metrics": metric_set_to_dict(validation_metrics),
            "test_metrics": metric_set_to_dict(test_metrics),
        }
        search_rows.append(search_row)
        if best_metrics is None or (
            validation_metrics.auc_pr,
            validation_metrics.auc_roc,
            -validation_metrics.brier_score,
        ) > (
            best_metrics.auc_pr,
            best_metrics.auc_roc,
            -best_metrics.brier_score,
        ):
            best_params = candidate_params
            best_metrics = validation_metrics

    if best_params is None or best_metrics is None:
        raise RuntimeError("No XGB_V1 candidate was evaluated successfully.")

    validation_model = OfficialXGBV1Model(best_params, seed=seed)
    validation_model.fit(train_X, train_y)
    validation_probabilities_raw = predict_probabilities(validation_model, validation_X)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(list(validation_probabilities_raw), validation_y)
    validation_probabilities_calibrated = [
        min(max(float(value), 0.0), 1.0) for value in calibrator.predict(list(validation_probabilities_raw))
    ]

    combined_frame = pd.concat([train_frame, validation_frame], ignore_index=True)
    combined_X = combined_frame.loc[:, OFFICIAL_INPUT_FEATURES]
    combined_y = [int(value) for value in combined_frame["target_defaulted"].tolist()]
    final_model = OfficialXGBV1Model(best_params, seed=seed)
    final_model.fit(combined_X, combined_y)
    test_probabilities_raw = predict_probabilities(final_model, test_X)
    test_probabilities_calibrated = [
        min(max(float(value), 0.0), 1.0) for value in calibrator.predict(list(test_probabilities_raw))
    ]

    OFFICIAL_MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OFFICIAL_MODEL_ARTIFACT_PATH.open("wb") as handle:
        pickle.dump(final_model, handle)
    with OFFICIAL_CALIBRATOR_ARTIFACT_PATH.open("wb") as handle:
        pickle.dump(calibrator, handle)

    feature_spec = {
        "model_version": OFFICIAL_MODEL_VERSION,
        "input_feature_columns": list(OFFICIAL_INPUT_FEATURES),
        "numeric_feature_columns": list(MODEL_NUMERIC_FEATURES),
        "categorical_feature_columns": list(MODEL_CATEGORICAL_FEATURES),
        "encoded_feature_names": final_model.get_feature_names(),
    }
    OFFICIAL_FEATURE_SPEC_PATH.write_text(json.dumps(feature_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_experiment_log(OFFICIAL_SEARCH_RESULTS_PATH, search_rows)
    top_feature_importance = extract_feature_importance(final_model, final_model.get_feature_names())[:25]

    summary = OfficialTrainingSummary(
        model_version=OFFICIAL_MODEL_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        data_path=str(data_path),
        data_hash=prepared_dataset.data_hash,
        split_counts=prepared_dataset.split_counts,
        selected_params=best_params,
        validation_metrics_raw=metric_set_to_dict(
            compute_binary_metrics(validation_y, validation_probabilities_raw)
        ),
        validation_metrics_calibrated=metric_set_to_dict(
            compute_binary_metrics(validation_y, validation_probabilities_calibrated)
        ),
        test_metrics_raw=metric_set_to_dict(compute_binary_metrics(test_y, test_probabilities_raw)),
        test_metrics_calibrated=metric_set_to_dict(
            compute_binary_metrics(test_y, test_probabilities_calibrated)
        ),
        top_feature_importance=top_feature_importance,
        model_artifact_path=str(OFFICIAL_MODEL_ARTIFACT_PATH),
        calibrator_artifact_path=str(OFFICIAL_CALIBRATOR_ARTIFACT_PATH),
        feature_spec_path=str(OFFICIAL_FEATURE_SPEC_PATH),
        manifest_path=str(OFFICIAL_MANIFEST_PATH),
        search_results_path=str(OFFICIAL_SEARCH_RESULTS_PATH),
    )
    OFFICIAL_MANIFEST_PATH.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _build_official_feature_chunk(raw_chunk):
    import pandas as pd

    frame = raw_chunk.copy()
    frame["application_type"] = _clean_string_series(frame["application_type"])
    frame["loan_status"] = _clean_string_series(frame["loan_status"])
    frame["issue_date"] = pd.to_datetime(frame["issue_d"], format="%b-%Y", errors="coerce")

    mask = (
        frame["application_type"].eq("Individual")
        & frame["issue_date"].notna()
        & frame["loan_status"].isin(DEFAULT_STATUS_MAP)
    )
    frame = frame.loc[mask].copy()
    if frame.empty:
        return frame

    annual_income = _numeric_series(frame["annual_inc"])
    loan_amount = _numeric_series(frame["loan_amnt"])
    term_months = _numeric_series(frame["term"].astype("string").str.extract(r"(\d+)")[0])
    valid_mask = annual_income.gt(0.0) & loan_amount.notna() & term_months.notna()
    frame = frame.loc[valid_mask].copy()
    if frame.empty:
        return frame

    annual_income = annual_income.loc[valid_mask].astype("float32")
    loan_amount = loan_amount.loc[valid_mask].astype("float32")
    term_months = term_months.loc[valid_mask].astype("float32")

    monthly_income = (annual_income / 12.0).astype("float32")
    dti_pct = _numeric_series(frame["dti"]).clip(lower=0.0).fillna(0.0).astype("float32")
    estimated_existing_emi = (monthly_income * (dti_pct / 100.0)).astype("float32")

    fico_low = _numeric_series(frame["fico_range_low"]).astype("float32")
    fico_high = _numeric_series(frame["fico_range_high"]).astype("float32")
    last_fico_low = _numeric_series(frame["last_fico_range_low"]).astype("float32")
    last_fico_high = _numeric_series(frame["last_fico_range_high"]).astype("float32")
    fico_midpoint = ((fico_low.fillna(0.0) + fico_high.fillna(0.0)) / 2.0).astype("float32")
    last_fico_midpoint = ((last_fico_low.fillna(0.0) + last_fico_high.fillna(0.0)) / 2.0).astype("float32")

    earliest_credit_line = pd.to_datetime(frame["earliest_cr_line"], format="%b-%Y", errors="coerce")
    credit_history_months = (
        (frame["issue_date"].dt.year - earliest_credit_line.dt.year) * 12
        + (frame["issue_date"].dt.month - earliest_credit_line.dt.month)
    ).fillna(0.0)
    credit_history_age_years = credit_history_months.clip(lower=0.0).astype("float32") / 12.0

    total_acc = _numeric_series(frame["total_acc"]).fillna(0.0).astype("float32")
    open_acc = _numeric_series(frame["open_acc"]).fillna(0.0).astype("float32")
    total_bc_limit = _numeric_series(frame["total_bc_limit"]).fillna(0.0).astype("float32")
    bc_util_ratio = _ratio_series(_numeric_series(frame["bc_util"]), 100.0).astype("float32")

    engineered = pd.DataFrame(
        {
            "loan_id": frame["id"].astype("string").fillna("").str.strip(),
            "issue_date": frame["issue_date"],
            "grade": _clean_string_series(frame["grade"]),
            "sub_grade": _clean_string_series(frame["sub_grade"]),
            "purpose": _clean_string_series(frame["purpose"]),
            "home_ownership": _clean_string_series(frame["home_ownership"]),
            "verification_status": _clean_string_series(frame["verification_status"]),
            "zip_code_prefix": _zip_code_prefix_series(frame),
            "employment_length_band": _employment_length_band_series(frame["emp_length"]),
            "loan_amount": loan_amount,
            "funded_amount": _numeric_series(frame["funded_amnt"]).fillna(0.0).astype("float32"),
            "term_months": term_months,
            "interest_rate_pct": _numeric_series(frame["int_rate"]).fillna(0.0).astype("float32"),
            "installment": _numeric_series(frame["installment"]).fillna(0.0).astype("float32"),
            "monthly_income": monthly_income,
            "estimated_existing_emi": estimated_existing_emi,
            "dti_ratio": _ratio_series(dti_pct, 100.0).astype("float32"),
            "loan_amount_to_income": _ratio_series(loan_amount, monthly_income).astype("float32"),
            "installment_to_income": _ratio_series(_numeric_series(frame["installment"]), monthly_income).astype("float32"),
            "existing_emi_to_income": _ratio_series(estimated_existing_emi, monthly_income).astype("float32"),
            "credit_score_midpoint": fico_midpoint,
            "credit_score_recent_delta": (last_fico_midpoint - fico_midpoint).astype("float32"),
            "credit_history_age_years": credit_history_age_years.astype("float32"),
            "employment_length_years": _employment_length_series(frame["emp_length"]).astype("float32"),
            "revol_util_ratio": _ratio_series(_numeric_series(frame["revol_util"]), 100.0).astype("float32"),
            "bc_util_ratio": bc_util_ratio,
            "all_util_ratio": _ratio_series(_numeric_series(frame["all_util"]), 100.0).astype("float32"),
            "il_util_ratio": _ratio_series(_numeric_series(frame["il_util"]), 100.0).astype("float32"),
            "revol_balance_to_income": _ratio_series(_numeric_series(frame["revol_bal"]), monthly_income).astype("float32"),
            "current_balance_to_income": _ratio_series(_numeric_series(frame["tot_cur_bal"]), monthly_income).astype("float32"),
            "total_balance_to_income": _ratio_series(_numeric_series(frame["total_bal_ex_mort"]), monthly_income).astype("float32"),
            "total_rev_limit_to_income": _ratio_series(_numeric_series(frame["total_rev_hi_lim"]), monthly_income).astype("float32"),
            "total_bc_limit_to_income": _ratio_series(total_bc_limit, monthly_income).astype("float32"),
            "credit_card_headroom_ratio": (1.0 - bc_util_ratio).clip(lower=0.0).astype("float32"),
            "delinquency_burden": _ratio_series(_numeric_series(frame["delinq_2yrs"]), total_acc).astype("float32"),
            "recent_inquiry_pressure": _ratio_series(_numeric_series(frame["inq_last_6mths"]), 6.0).astype("float32"),
            "credit_inquiry_velocity": _ratio_series(_numeric_series(frame["inq_last_12m"]), 12.0).astype("float32"),
            "open_account_density": _ratio_series(open_acc, total_acc).astype("float32"),
            "accounts_per_year": _ratio_series(total_acc, credit_history_age_years).astype("float32"),
            "balance_per_open_account": _ratio_series(_numeric_series(frame["tot_cur_bal"]), open_acc).astype("float32"),
            "mortgage_account_share": _ratio_series(_numeric_series(frame["mort_acc"]), total_acc).astype("float32"),
            "bankruptcy_flag": _numeric_series(frame["pub_rec_bankruptcies"]).fillna(0.0).gt(0.0).astype("float32"),
            "tax_lien_flag": _numeric_series(frame["tax_liens"]).fillna(0.0).gt(0.0).astype("float32"),
            "high_utilization_fraction": _ratio_series(_numeric_series(frame["percent_bc_gt_75"]), 100.0).astype("float32"),
            "never_delinquent_ratio": _ratio_series(_numeric_series(frame["pct_tl_nvr_dlq"]), 100.0).astype("float32"),
            "collections_12m": _numeric_series(frame["collections_12_mths_ex_med"]).fillna(0.0).astype("float32"),
            "recent_revolving_trade_gap_months": _numeric_series(frame["mo_sin_rcnt_rev_tl_op"]).fillna(0.0).astype("float32"),
            "revolving_trade_age_years": _ratio_series(_numeric_series(frame["mo_sin_old_rev_tl_op"]), 12.0).astype("float32"),
            "open_revolving_24m": _numeric_series(frame["open_rv_24m"]).fillna(0.0).astype("float32"),
            "open_installment_24m": _numeric_series(frame["open_il_24m"]).fillna(0.0).astype("float32"),
            "target_defaulted": frame["loan_status"].map(DEFAULT_STATUS_MAP).astype("int8"),
        }
    )
    for feature_name in MODEL_CATEGORICAL_FEATURES:
        engineered[feature_name] = engineered[feature_name].astype("category")
    return engineered


def _official_split_masks(feature_frame):
    issue_year = feature_frame["issue_date"].dt.year
    return {
        "train": issue_year.le(2016),
        "validation": issue_year.eq(2017),
        "test": issue_year.eq(2018),
        "holdout": issue_year.gt(2018),
    }


def _clean_string_series(series):
    return series.astype("string").fillna("").str.strip().replace("", "UNKNOWN")


def _available_csv_columns(data_path):
    import pandas as pd

    preview = pd.read_csv(data_path, nrows=0)
    if hasattr(preview, "columns"):
        return set(preview.columns.tolist())
    if isinstance(preview, list) and preview and hasattr(preview[0], "columns"):
        return set(preview[0].columns.tolist())
    return set(MODELING_COLUMNS)


def _numeric_series(series):
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")


def _ratio_series(numerator, denominator):
    numerator_values = _numeric_series(numerator).fillna(0.0).astype("float32")
    if isinstance(denominator, (int, float)):
        denominator_values = float(denominator)
        if denominator_values == 0.0:
            return numerator_values * 0.0
        result = numerator_values / denominator_values
        return result.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    denominator_values = _numeric_series(denominator).fillna(0.0).astype("float32")
    result = numerator_values.divide(denominator_values.where(denominator_values != 0.0, other=1.0))
    return result.where(denominator_values != 0.0, other=0.0).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)


def _employment_length_series(series):
    cleaned = series.astype("string").fillna("").str.strip()
    mapped = cleaned.replace(
        {
            "< 1 year": "0.5",
            "10+ years": "10",
        },
        regex=False,
    )
    extracted = mapped.str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0]
    return _numeric_series(extracted).fillna(0.0)


def _employment_length_band_series(series):
    import pandas as pd

    numeric_years = _employment_length_series(series)
    banded = []
    for value in numeric_years.tolist():
        years = float(value)
        if years <= 0.0:
            banded.append("0")
        elif years < 3.0:
            banded.append("1-2")
        elif years < 6.0:
            banded.append("3-5")
        elif years < 10.0:
            banded.append("6-9")
        else:
            banded.append("10+")
    return pd.Series(banded, index=numeric_years.index, dtype="string").fillna("UNKNOWN")


def _zip_code_prefix_series(frame):
    import pandas as pd

    if "zip_code" not in frame.columns:
        return pd.Series(["UNKNOWN"] * len(frame), index=frame.index, dtype="string")
    cleaned = frame["zip_code"].astype("string").fillna("").str.strip()
    prefixes = cleaned.str.extract(r"([0-9]{3})", expand=False).fillna("UNKNOWN")
    return prefixes.astype("string")


def _deterministic_search_slice(frame, *, max_rows: int):
    if len(frame) <= max_rows:
        return frame
    stride = max(len(frame) // max_rows, 1)
    sliced = frame.iloc[::stride].head(max_rows).copy()
    if len(sliced) < max_rows:
        return frame.head(max_rows).copy()
    return sliced


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for `python -m ml.models.train`."""

    parser = argparse.ArgumentParser(description="Train Phase 3 AuditLend ML candidates.")
    parser.add_argument("--official-xgb-v1", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--modulo-sampling", type=int, default=1)
    parser.add_argument("--search-metric", default=DEFAULT_SEARCH_METRIC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-label", default="phase3")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-lightgbm", action="store_true")
    parser.add_argument("--skip-logistic-regression", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the Phase 3 trainer."""

    args = parse_args(argv)
    if args.official_xgb_v1:
        summary = train_official_xgb_v1(seed=args.seed)
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
        return 0
    config = TrainingConfig(
        seed=args.seed,
        search_metric=args.search_metric,
        max_rows_per_split=args.max_rows_per_split,
        modulo_sampling=max(args.modulo_sampling, 1),
        run_label=args.run_label,
        include_xgboost=not args.skip_xgboost,
        include_lightgbm=not args.skip_lightgbm,
        include_logistic_regression=not args.skip_logistic_regression,
    )
    summary = run_training(config)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


def _build_split_feature_rows(
    *,
    max_rows_per_split: int | None,
    modulo_sampling: int,
    env_var: str,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
        "holdout": [],
    }

    for clean_row in iter_clean_lending_club_rows(env_var=env_var):
        feature_row = build_feature_row(clean_row)
        split_name = assign_time_split(feature_row["issue_date"])
        if max_rows_per_split is not None and len(buckets[split_name]) >= max_rows_per_split:
            continue
        if not should_keep_row(feature_row.get("loan_id"), modulo_sampling):
            continue
        buckets[split_name].append(feature_row)
        if max_rows_per_split is not None and all(
            len(buckets[name]) >= max_rows_per_split for name in ("train", "validation", "test")
        ):
            break

    return buckets


def should_keep_row(loan_id: Any, modulo_sampling: int) -> bool:
    """Deterministic row subsampling by hashed loan id."""

    if modulo_sampling <= 1:
        return True
    if loan_id is None:
        return False
    raw_id = str(loan_id)
    if not raw_id:
        return False
    return int(sha256(raw_id.encode("utf-8")).hexdigest(), 16) % modulo_sampling == 0


def normalize_category_value(value: Any) -> str:
    raw_value = str(value or "").strip()
    return raw_value if raw_value else "UNKNOWN"


def _as_model_input(
    X: Sequence[Sequence[float]],
    feature_names: Sequence[str] | None = None,
):
    if hasattr(X, "loc"):
        if feature_names is None:
            return X
        return X.loc[:, list(feature_names)]
    if feature_names is None:
        return list(X)
    try:
        import pandas as pd
    except ImportError:
        return list(X)
    return pd.DataFrame(list(X), columns=list(feature_names))


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _expand_grid(grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = sorted(grid.keys())
    combinations = product(*(grid[key] for key in keys))
    return [{key: value for key, value in zip(keys, combination)} for combination in combinations]


def _sigmoid(score: float) -> float:
    bounded_score = max(min(score, 60.0), -60.0)
    return 1.0 / (1.0 + exp(-bounded_score))


def _build_logistic_regression_factory(seed: int) -> Callable[[dict[str, Any]], Any] | None:
    if importlib.util.find_spec("sklearn.linear_model") is None:
        return None

    def factory(params: dict[str, Any]) -> Any:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=seed, **params)),
            ]
        )

    return factory


def _build_xgboost_factory(seed: int) -> Callable[[dict[str, Any]], Any] | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    def factory(params: dict[str, Any]) -> Any:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=4,
            tree_method="hist",
            verbosity=0,
            **params,
        )

    return factory


def _build_lightgbm_factory(seed: int) -> Callable[[dict[str, Any]], Any] | None:
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None

    def factory(params: dict[str, Any]) -> Any:
        return LGBMClassifier(random_state=seed, n_jobs=4, objective="binary", verbose=-1, **params)

    return factory


def _compute_auc_roc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    positive_count = sum(1 for target in y_true if int(target) == 1)
    negative_count = len(y_true) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0.5

    ranked = sorted(zip(scores, y_true), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        next_index = index + 1
        while next_index < len(ranked) and ranked[next_index][0] == ranked[index][0]:
            next_index += 1
        average_rank = (index + 1 + next_index) / 2.0
        positives_in_group = sum(1 for _, target in ranked[index:next_index] if int(target) == 1)
        rank_sum += positives_in_group * average_rank
        index = next_index

    auc = (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    return float(auc)


def _compute_average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(1 for target in y_true if int(target) == 1)
    if positives == 0:
        return 0.0

    ranked = sorted(zip(scores, y_true), key=lambda pair: pair[0], reverse=True)
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0

    for _, target in ranked:
        if int(target) == 1:
            true_positives += 1
        else:
            false_positives += 1
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += precision * max(recall - previous_recall, 0.0)
        previous_recall = recall

    return float(area)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
