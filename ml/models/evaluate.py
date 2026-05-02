"""Evaluation, selection, and reporting utilities for Phase 4 model review."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from ml.models.train import (
    MetricSet,
    OFFICIAL_INPUT_FEATURES,
    OFFICIAL_MANIFEST_PATH,
    OFFICIAL_MODEL_VERSION,
    PreparedDataset,
    TrainingConfig,
    compute_binary_metrics,
    metric_set_to_dict,
    predict_probabilities,
    prepare_official_training_dataset,
    prepare_training_dataset,
)

DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_ECE_BINS = 10


@dataclass(frozen=True)
class CalibrationSummary:
    """Expected calibration error summary for a split."""

    ece: float
    max_calibration_gap: float
    bins: list[dict[str, float | int]]


@dataclass(frozen=True)
class ConfusionMatrixSummary:
    """Confusion matrix counts and derived metrics for one threshold."""

    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    specificity: float
    false_positive_rate: float
    accuracy: float


@dataclass(frozen=True)
class SegmentSummary:
    """Segment-level diagnostics on the evaluation split."""

    segment_field: str
    segment_value: str
    row_count: int
    positive_rate: float
    mean_score: float
    predicted_positive_rate: float


@dataclass(frozen=True)
class EvaluationSplitSummary:
    """Model metrics and diagnostics for one split."""

    name: str
    metrics: MetricSet
    calibration: CalibrationSummary
    thresholds: list[ConfusionMatrixSummary]


@dataclass(frozen=True)
class CandidateComparisonRow:
    """Best-known configuration for one candidate family from the search log."""

    candidate_name: str
    family: str
    params: dict[str, Any]
    validation_auc_pr: float
    validation_auc_roc: float
    validation_brier_score: float
    test_auc_pr: float
    test_auc_roc: float
    test_brier_score: float


@dataclass(frozen=True)
class EvaluationReport:
    """Serializable result of a Phase 4 evaluation run."""

    run_id: str
    manifest_path: str
    artifact_path: str
    search_results_path: str
    selected_candidate: str
    selected_params: dict[str, Any]
    feature_count: int
    split_counts: dict[str, int]
    evaluation_splits: dict[str, dict[str, Any]]
    candidate_comparison: list[dict[str, Any]]
    top_feature_importance: list[dict[str, float]]
    segment_diagnostics: list[dict[str, Any]]
    report_path: str


@dataclass(frozen=True)
class OfficialEvaluationReport:
    """Evaluation summary for the signed-off XGB_V1 artifact set."""

    model_version: str
    manifest_path: str
    report_path: str
    split_counts: dict[str, int]
    selected_params: dict[str, Any]
    raw_splits: dict[str, dict[str, Any]]
    calibrated_splits: dict[str, dict[str, Any]]
    top_feature_importance: list[dict[str, float]]


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load a Phase 3 manifest from disk."""

    path = Path(manifest_path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_artifact(artifact_path: str | Path) -> Any:
    """Load a pickled trained estimator."""

    with Path(artifact_path).open("rb") as handle:
        return pickle.load(handle)


def load_search_results(search_results_path: str | Path) -> list[dict[str, Any]]:
    """Load search results JSONL rows."""

    rows: list[dict[str, Any]] = []
    with Path(search_results_path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def compute_expected_calibration_error(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    bins: int = DEFAULT_ECE_BINS,
) -> CalibrationSummary:
    """Compute expected calibration error with equal-width bins."""

    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length.")
    if bins <= 0:
        raise ValueError("bins must be positive.")
    if not y_true:
        return CalibrationSummary(ece=0.0, max_calibration_gap=0.0, bins=[])

    bucket_rows: list[list[tuple[int, float]]] = [[] for _ in range(bins)]
    for target, probability in zip(y_true, probabilities, strict=True):
        clipped_probability = min(max(float(probability), 0.0), 1.0)
        bucket_index = min(int(clipped_probability * bins), bins - 1)
        bucket_rows[bucket_index].append((int(target), clipped_probability))

    ece = 0.0
    max_gap = 0.0
    summaries: list[dict[str, float | int]] = []

    for index, bucket in enumerate(bucket_rows):
        lower_bound = index / bins
        upper_bound = (index + 1) / bins
        if not bucket:
            summaries.append(
                {
                    "bin_index": index,
                    "lower_bound": round(lower_bound, 4),
                    "upper_bound": round(upper_bound, 4),
                    "count": 0,
                    "mean_prediction": 0.0,
                    "observed_default_rate": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        mean_prediction = mean(probability for _, probability in bucket)
        observed_default_rate = mean(target for target, _ in bucket)
        gap = abs(mean_prediction - observed_default_rate)
        weight = len(bucket) / len(y_true)
        ece += gap * weight
        max_gap = max(max_gap, gap)
        summaries.append(
            {
                "bin_index": index,
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "count": len(bucket),
                "mean_prediction": round(mean_prediction, 6),
                "observed_default_rate": round(observed_default_rate, 6),
                "gap": round(gap, 6),
            }
        )

    return CalibrationSummary(
        ece=round(ece, 6),
        max_calibration_gap=round(max_gap, 6),
        bins=summaries,
    )


def compute_confusion_matrix_summary(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> ConfusionMatrixSummary:
    """Compute thresholded classification diagnostics."""

    tp = fp = tn = fn = 0
    for target, probability in zip(y_true, probabilities, strict=True):
        prediction = 1 if float(probability) >= threshold else 0
        if prediction == 1 and int(target) == 1:
            tp += 1
        elif prediction == 1 and int(target) == 0:
            fp += 1
        elif prediction == 0 and int(target) == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return ConfusionMatrixSummary(
        threshold=threshold,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        precision=round(precision, 6),
        recall=round(recall, 6),
        specificity=round(specificity, 6),
        false_positive_rate=round(false_positive_rate, 6),
        accuracy=round(accuracy, 6),
    )


def evaluate_split(
    split_name: str,
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> EvaluationSplitSummary:
    """Evaluate one split with metrics, calibration, and threshold analysis."""

    metrics = compute_binary_metrics(y_true, probabilities)
    calibration = compute_expected_calibration_error(y_true, probabilities, bins=ece_bins)
    threshold_rows = [
        compute_confusion_matrix_summary(y_true, probabilities, threshold=threshold)
        for threshold in thresholds
    ]
    return EvaluationSplitSummary(
        name=split_name,
        metrics=metrics,
        calibration=calibration,
        thresholds=threshold_rows,
    )


def summarize_candidate_comparison(search_rows: Sequence[dict[str, Any]]) -> list[CandidateComparisonRow]:
    """Pick the best validation AUC-PR row per candidate family."""

    best_by_candidate: dict[str, dict[str, Any]] = {}
    for row in search_rows:
        candidate_name = str(row["candidate_name"])
        current = best_by_candidate.get(candidate_name)
        if current is None or (
            float(row["validation_metrics"]["auc_pr"]),
            float(row["validation_metrics"]["auc_roc"]),
            -float(row["validation_metrics"]["brier_score"]),
        ) > (
            float(current["validation_metrics"]["auc_pr"]),
            float(current["validation_metrics"]["auc_roc"]),
            -float(current["validation_metrics"]["brier_score"]),
        ):
            best_by_candidate[candidate_name] = row

    comparison_rows = [
        CandidateComparisonRow(
            candidate_name=candidate_name,
            family=str(row["family"]),
            params=dict(row["params"]),
            validation_auc_pr=float(row["validation_metrics"]["auc_pr"]),
            validation_auc_roc=float(row["validation_metrics"]["auc_roc"]),
            validation_brier_score=float(row["validation_metrics"]["brier_score"]),
            test_auc_pr=float(row["test_metrics"]["auc_pr"]),
            test_auc_roc=float(row["test_metrics"]["auc_roc"]),
            test_brier_score=float(row["test_metrics"]["brier_score"]),
        )
        for candidate_name, row in best_by_candidate.items()
    ]
    return sorted(
        comparison_rows,
        key=lambda row: (row.validation_auc_pr, row.validation_auc_roc, -row.validation_brier_score),
        reverse=True,
    )


def compute_segment_diagnostics(
    rows: Sequence[dict[str, Any]],
    probabilities: Sequence[float],
    *,
    segment_fields: Sequence[str] = ("home_ownership", "verification_status"),
    threshold: float = 0.5,
    min_count: int = 25,
) -> list[SegmentSummary]:
    """Compute simple segment-level score and prediction summaries."""

    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        for segment_field in segment_fields:
            segment_value = str(row.get(segment_field, "UNKNOWN") or "UNKNOWN")
            grouped[(segment_field, segment_value)].append((int(row.get("target_defaulted", 0)), float(probability)))

    summaries: list[SegmentSummary] = []
    for (segment_field, segment_value), values in grouped.items():
        if len(values) < min_count:
            continue
        positive_rate = mean(target for target, _ in values)
        mean_score = mean(probability for _, probability in values)
        predicted_positive_rate = mean(1 if probability >= threshold else 0 for _, probability in values)
        summaries.append(
            SegmentSummary(
                segment_field=segment_field,
                segment_value=segment_value,
                row_count=len(values),
                positive_rate=round(positive_rate, 6),
                mean_score=round(mean_score, 6),
                predicted_positive_rate=round(predicted_positive_rate, 6),
            )
        )

    return sorted(summaries, key=lambda summary: (summary.segment_field, -summary.row_count, summary.segment_value))


def evaluate_manifest(
    manifest_path: str | Path,
    *,
    max_rows_per_split: int | None = None,
    modulo_sampling: int = 1,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    report_dir: str | Path = "ml/models/reports",
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> EvaluationReport:
    """Evaluate one training manifest and write a markdown report."""

    manifest = load_manifest(manifest_path)
    model = load_model_artifact(manifest["artifact_path"])
    search_rows = load_search_results(manifest["experiment_log_path"])
    prepared_dataset = prepare_training_dataset(
        max_rows_per_split=max_rows_per_split,
        modulo_sampling=modulo_sampling,
        env_var=env_var,
    )

    split_summaries: dict[str, EvaluationSplitSummary] = {}
    split_probabilities: dict[str, list[float]] = {}
    for split_name in ("train", "validation", "test"):
        split_matrix = prepared_dataset.split_matrices[split_name]
        probabilities = predict_probabilities(model, split_matrix.X, feature_names=split_matrix.feature_names)
        split_probabilities[split_name] = probabilities
        split_summaries[split_name] = evaluate_split(
            split_name,
            split_matrix.y,
            probabilities,
            thresholds=thresholds,
            ece_bins=ece_bins,
        )

    candidate_comparison = summarize_candidate_comparison(search_rows)
    test_split = prepared_dataset.split_matrices["test"]
    segment_diagnostics = compute_segment_diagnostics(test_split.rows, split_probabilities["test"])
    report_dir_path = Path(report_dir)
    report_dir_path.mkdir(parents=True, exist_ok=True)
    report_path = report_dir_path / f"{manifest['run_id']}_evaluation.md"

    evaluation_report = EvaluationReport(
        run_id=str(manifest["run_id"]),
        manifest_path=str(Path(manifest_path)),
        artifact_path=str(manifest["artifact_path"]),
        search_results_path=str(manifest["experiment_log_path"]),
        selected_candidate=str(manifest["selected_candidate"]),
        selected_params=dict(manifest["selected_params"]),
        feature_count=int(manifest["feature_count"]),
        split_counts={name: int(count) for name, count in manifest["split_counts"].items()},
        evaluation_splits={
            split_name: evaluation_split_to_dict(summary)
            for split_name, summary in split_summaries.items()
        },
        candidate_comparison=[asdict(row) for row in candidate_comparison],
        top_feature_importance=list(manifest.get("feature_importance", [])),
        segment_diagnostics=[asdict(summary) for summary in segment_diagnostics],
        report_path=str(report_path),
    )
    write_evaluation_report(report_path, evaluation_report)
    return evaluation_report


def write_evaluation_report(output_path: str | Path, evaluation_report: EvaluationReport) -> Path:
    """Write a markdown evaluation report for human review."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = evaluation_report.evaluation_splits
    candidate_rows = evaluation_report.candidate_comparison
    segment_rows = evaluation_report.segment_diagnostics[:12]

    lines = [
        "# Phase 4 Model Evaluation Report",
        "",
        f"Run ID: `{evaluation_report.run_id}`",
        f"Selected candidate: `{evaluation_report.selected_candidate}`",
        f"Artifact: `{evaluation_report.artifact_path}`",
        "",
        "## Selection Summary",
        "",
        f"- Search metric: validation AUC-PR",
        f"- Feature count: {evaluation_report.feature_count}",
        f"- Split counts: {evaluation_report.split_counts}",
        f"- Selected params: `{json.dumps(evaluation_report.selected_params, sort_keys=True)}`",
        "",
        "## Split Metrics",
        "",
        "| Split | AUC-ROC | AUC-PR | Brier | ECE | Max Gap | Rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for split_name in ("train", "validation", "test"):
        split_summary = splits[split_name]
        metrics = split_summary["metrics"]
        calibration = split_summary["calibration"]
        lines.append(
            "| "
            f"{split_name} | {metrics['auc_roc']:.6f} | {metrics['auc_pr']:.6f} | "
            f"{metrics['brier_score']:.6f} | {calibration['ece']:.6f} | "
            f"{calibration['max_calibration_gap']:.6f} | {metrics['row_count']} |"
        )

    lines.extend(
        [
            "",
            "## Candidate Comparison",
            "",
            "| Candidate | Family | Val AUC-PR | Val AUC-ROC | Val Brier | Test AUC-PR | Test AUC-ROC | Test Brier |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in candidate_rows:
        lines.append(
            "| "
            f"{row['candidate_name']} | {row['family']} | {row['validation_auc_pr']:.6f} | "
            f"{row['validation_auc_roc']:.6f} | {row['validation_brier_score']:.6f} | "
            f"{row['test_auc_pr']:.6f} | {row['test_auc_roc']:.6f} | {row['test_brier_score']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Threshold Analysis (Test Split)",
            "",
            "| Threshold | TP | FP | TN | FN | Precision | Recall | Specificity | FPR | Accuracy |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in splits["test"]["thresholds"]:
        lines.append(
            "| "
            f"{row['threshold']:.2f} | {row['true_positive']} | {row['false_positive']} | "
            f"{row['true_negative']} | {row['false_negative']} | {row['precision']:.6f} | "
            f"{row['recall']:.6f} | {row['specificity']:.6f} | {row['false_positive_rate']:.6f} | "
            f"{row['accuracy']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Top Feature Importance",
            "",
            "| Feature | Importance |",
            "| --- | ---: |",
        ]
    )
    for row in evaluation_report.top_feature_importance[:15]:
        lines.append(f"| {row['feature']} | {row['importance']:.8f} |")

    if segment_rows:
        lines.extend(
            [
                "",
                "## Segment Diagnostics",
                "",
                "| Segment | Value | Rows | Positive Rate | Mean Score | Predicted Positive Rate |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in segment_rows:
            lines.append(
                "| "
                f"{row['segment_field']} | {row['segment_value']} | {row['row_count']} | "
                f"{row['positive_rate']:.6f} | {row['mean_score']:.6f} | {row['predicted_positive_rate']:.6f} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def evaluation_split_to_dict(summary: EvaluationSplitSummary) -> dict[str, Any]:
    """Serialize one split summary."""

    return {
        "name": summary.name,
        "metrics": metric_set_to_dict(summary.metrics),
        "calibration": asdict(summary.calibration),
        "thresholds": [asdict(row) for row in summary.thresholds],
    }


def find_latest_manifest(experiment_dir: str | Path = "ml/models/experiments") -> Path:
    """Return the most recently named manifest in the experiment directory."""

    manifest_paths = sorted(Path(experiment_dir).glob("*/manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError("No manifest.json files found under ml/models/experiments.")
    return manifest_paths[-1]


def evaluate_official_xgb_v1(
    *,
    manifest_path: str | Path = OFFICIAL_MANIFEST_PATH,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    report_dir: str | Path = "ml/models/reports",
    ece_bins: int = DEFAULT_ECE_BINS,
) -> OfficialEvaluationReport:
    """Evaluate the official XGB_V1 artifact set on the PRD-aligned splits."""

    manifest = load_manifest(manifest_path)
    model = load_model_artifact(manifest["model_artifact_path"])
    calibrator = load_model_artifact(manifest["calibrator_artifact_path"])
    prepared_dataset = prepare_official_training_dataset(env_var=env_var)

    raw_summaries: dict[str, dict[str, Any]] = {}
    calibrated_summaries: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation", "test"):
        split_frame = prepared_dataset.split_frames[split_name]
        probabilities_raw = predict_probabilities(model, split_frame.loc[:, OFFICIAL_INPUT_FEATURES])
        probabilities_calibrated = [
            min(max(float(value), 0.0), 1.0) for value in calibrator.predict(list(probabilities_raw))
        ]
        y_true = [int(value) for value in split_frame["target_defaulted"].tolist()]
        raw_summaries[split_name] = {
            "metrics": metric_set_to_dict(compute_binary_metrics(y_true, probabilities_raw)),
            "calibration": asdict(compute_expected_calibration_error(y_true, probabilities_raw, bins=ece_bins)),
        }
        calibrated_summaries[split_name] = {
            "metrics": metric_set_to_dict(compute_binary_metrics(y_true, probabilities_calibrated)),
            "calibration": asdict(compute_expected_calibration_error(y_true, probabilities_calibrated, bins=ece_bins)),
        }

    report_dir_path = Path(report_dir)
    report_dir_path.mkdir(parents=True, exist_ok=True)
    report_path = report_dir_path / f"{OFFICIAL_MODEL_VERSION}_evaluation.md"
    report = OfficialEvaluationReport(
        model_version=str(manifest.get("model_version", OFFICIAL_MODEL_VERSION)),
        manifest_path=str(Path(manifest_path)),
        report_path=str(report_path),
        split_counts={name: int(count) for name, count in manifest["split_counts"].items()},
        selected_params=dict(manifest["selected_params"]),
        raw_splits=raw_summaries,
        calibrated_splits=calibrated_summaries,
        top_feature_importance=list(manifest.get("top_feature_importance", [])),
    )
    write_official_evaluation_report(report_path, report)
    return report


def write_official_evaluation_report(output_path: str | Path, report: OfficialEvaluationReport) -> Path:
    """Write a markdown evaluation report for the official XGB_V1 model."""

    lines = [
        "# Official XGB_V1 Evaluation Report",
        "",
        f"Model version: `{report.model_version}`",
        f"Manifest: `{report.manifest_path}`",
        "",
        "## Split Metrics",
        "",
        "| Split | Variant | AUC-ROC | AUC-PR | Brier | ECE | Max Gap | Rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ("train", "validation", "test"):
        for variant_name, split_map in (("raw", report.raw_splits), ("calibrated", report.calibrated_splits)):
            metrics = split_map[split_name]["metrics"]
            calibration = split_map[split_name]["calibration"]
            lines.append(
                f"| {split_name} | {variant_name} | {metrics['auc_roc']:.6f} | {metrics['auc_pr']:.6f} | "
                f"{metrics['brier_score']:.6f} | {calibration['ece']:.6f} | {calibration['max_calibration_gap']:.6f} | "
                f"{metrics['row_count']} |"
            )

    lines.extend(
        [
            "",
            "## Selected Parameters",
            "",
            f"`{json.dumps(report.selected_params, sort_keys=True)}`",
            "",
            "## Top Feature Importance",
            "",
            "| Feature | Importance |",
            "| --- | ---: |",
        ]
    )
    for row in report.top_feature_importance[:15]:
        lines.append(f"| {row['feature']} | {row['importance']:.8f} |")

    path = Path(output_path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for `python -m ml.models.evaluate`."""

    parser = argparse.ArgumentParser(description="Evaluate a trained Phase 3 model artifact.")
    parser.add_argument("--official-xgb-v1", action="store_true")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--experiment-dir", default="ml/models/experiments")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--modulo-sampling", type=int, default=1)
    parser.add_argument("--report-dir", default="ml/models/reports")
    parser.add_argument("--ece-bins", type=int, default=DEFAULT_ECE_BINS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for evaluation."""

    args = parse_args(argv)
    if args.official_xgb_v1:
        report = evaluate_official_xgb_v1(
            manifest_path=Path(args.manifest_path) if args.manifest_path else OFFICIAL_MANIFEST_PATH,
            report_dir=args.report_dir,
            ece_bins=args.ece_bins,
        )
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0
    manifest_path = Path(args.manifest_path) if args.manifest_path else find_latest_manifest(args.experiment_dir)
    report = evaluate_manifest(
        manifest_path,
        max_rows_per_split=args.max_rows_per_split,
        modulo_sampling=max(args.modulo_sampling, 1),
        report_dir=args.report_dir,
        ece_bins=args.ece_bins,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
