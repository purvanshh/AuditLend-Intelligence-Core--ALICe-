"""Probability calibration workflow for Phase 5 model refinement."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from ml.models.evaluate import (
    CalibrationSummary,
    compute_expected_calibration_error,
    find_latest_manifest,
    load_manifest,
    load_model_artifact,
)
from ml.models.train import (
    MetricSet,
    compute_binary_metrics,
    metric_set_to_dict,
    predict_probabilities,
    prepare_training_dataset,
)

DEFAULT_CALIBRATION_BINS = 10


@dataclass(frozen=True)
class CalibrationSplitSummary:
    """Before/after calibration metrics for one split."""

    name: str
    raw_metrics: MetricSet
    calibrated_metrics: MetricSet
    raw_calibration: CalibrationSummary
    calibrated_calibration: CalibrationSummary


@dataclass(frozen=True)
class CalibrationRunSummary:
    """Serializable summary for a Phase 5 calibration run."""

    run_id: str
    manifest_path: str
    model_artifact_path: str
    calibrator_artifact_path: str
    calibration_manifest_path: str
    report_path: str
    validation_curve_svg_path: str
    test_curve_svg_path: str
    selected_candidate: str
    split_counts: dict[str, int]
    splits: dict[str, dict[str, Any]]


def fit_isotonic_calibrator(
    y_true: Sequence[int],
    probabilities: Sequence[float],
):
    """Fit an isotonic regression calibrator on validation predictions."""

    from sklearn.isotonic import IsotonicRegression

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(list(probabilities), list(y_true))
    return calibrator


def apply_calibrator(calibrator: Any, probabilities: Sequence[float]) -> list[float]:
    """Apply a fitted calibrator and clip outputs into [0, 1]."""

    calibrated = calibrator.predict(list(probabilities))
    return [min(max(float(value), 0.0), 1.0) for value in calibrated]


def summarize_calibrated_split(
    split_name: str,
    y_true: Sequence[int],
    raw_probabilities: Sequence[float],
    calibrated_probabilities: Sequence[float],
    *,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> CalibrationSplitSummary:
    """Build before/after metrics and calibration summaries for one split."""

    raw_metrics = compute_binary_metrics(y_true, raw_probabilities)
    calibrated_metrics = compute_binary_metrics(y_true, calibrated_probabilities)
    raw_calibration = compute_expected_calibration_error(y_true, raw_probabilities, bins=bins)
    calibrated_calibration = compute_expected_calibration_error(y_true, calibrated_probabilities, bins=bins)
    return CalibrationSplitSummary(
        name=split_name,
        raw_metrics=raw_metrics,
        calibrated_metrics=calibrated_metrics,
        raw_calibration=raw_calibration,
        calibrated_calibration=calibrated_calibration,
    )


def calibrate_manifest(
    manifest_path: str | Path,
    *,
    max_rows_per_split: int | None = None,
    modulo_sampling: int = 1,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    bins: int = DEFAULT_CALIBRATION_BINS,
    report_dir: str | Path = "ml/models/reports",
) -> CalibrationRunSummary:
    """Fit and persist an isotonic calibrator for one training manifest."""

    manifest = load_manifest(manifest_path)
    model = load_model_artifact(manifest["artifact_path"])
    prepared_dataset = prepare_training_dataset(
        max_rows_per_split=max_rows_per_split,
        modulo_sampling=modulo_sampling,
        env_var=env_var,
    )

    validation_split = prepared_dataset.split_matrices["validation"]
    test_split = prepared_dataset.split_matrices["test"]

    raw_validation_probabilities = predict_probabilities(
        model,
        validation_split.X,
        feature_names=validation_split.feature_names,
    )
    raw_test_probabilities = predict_probabilities(
        model,
        test_split.X,
        feature_names=test_split.feature_names,
    )

    calibrator = fit_isotonic_calibrator(validation_split.y, raw_validation_probabilities)
    calibrated_validation_probabilities = apply_calibrator(calibrator, raw_validation_probabilities)
    calibrated_test_probabilities = apply_calibrator(calibrator, raw_test_probabilities)

    validation_summary = summarize_calibrated_split(
        "validation",
        validation_split.y,
        raw_validation_probabilities,
        calibrated_validation_probabilities,
        bins=bins,
    )
    test_summary = summarize_calibrated_split(
        "test",
        test_split.y,
        raw_test_probabilities,
        calibrated_test_probabilities,
        bins=bins,
    )

    artifact_path = Path(manifest["artifact_path"])
    artifact_dir = artifact_path.parent
    calibrator_artifact_path = artifact_dir / "isotonic_calibrator.pkl"
    with calibrator_artifact_path.open("wb") as handle:
        pickle.dump(calibrator, handle)

    report_dir_path = Path(report_dir)
    report_dir_path.mkdir(parents=True, exist_ok=True)
    validation_curve_svg_path = report_dir_path / f"{manifest['run_id']}_validation_calibration.svg"
    test_curve_svg_path = report_dir_path / f"{manifest['run_id']}_test_calibration.svg"
    report_path = report_dir_path / f"{manifest['run_id']}_calibration.md"

    write_calibration_curve_svg(
        validation_curve_svg_path,
        title=f"{manifest['run_id']} validation calibration",
        raw_calibration=validation_summary.raw_calibration,
        calibrated_calibration=validation_summary.calibrated_calibration,
    )
    write_calibration_curve_svg(
        test_curve_svg_path,
        title=f"{manifest['run_id']} test calibration",
        raw_calibration=test_summary.raw_calibration,
        calibrated_calibration=test_summary.calibrated_calibration,
    )

    calibration_manifest_path = artifact_dir / "isotonic_calibration_manifest.json"
    summary = CalibrationRunSummary(
        run_id=str(manifest["run_id"]),
        manifest_path=str(Path(manifest_path)),
        model_artifact_path=str(artifact_path),
        calibrator_artifact_path=str(calibrator_artifact_path),
        calibration_manifest_path=str(calibration_manifest_path),
        report_path=str(report_path),
        validation_curve_svg_path=str(validation_curve_svg_path),
        test_curve_svg_path=str(test_curve_svg_path),
        selected_candidate=str(manifest["selected_candidate"]),
        split_counts={name: int(count) for name, count in manifest["split_counts"].items()},
        splits={
            "validation": calibration_split_to_dict(validation_summary),
            "test": calibration_split_to_dict(test_summary),
        },
    )
    calibration_manifest_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_calibration_report(report_path, summary)
    return summary


def calibration_split_to_dict(summary: CalibrationSplitSummary) -> dict[str, Any]:
    """Serialize a calibration split summary."""

    return {
        "name": summary.name,
        "raw_metrics": metric_set_to_dict(summary.raw_metrics),
        "calibrated_metrics": metric_set_to_dict(summary.calibrated_metrics),
        "raw_calibration": asdict(summary.raw_calibration),
        "calibrated_calibration": asdict(summary.calibrated_calibration),
    }


def write_calibration_curve_svg(
    output_path: str | Path,
    *,
    title: str,
    raw_calibration: CalibrationSummary,
    calibrated_calibration: CalibrationSummary,
) -> Path:
    """Render a simple SVG reliability curve with raw and calibrated lines."""

    width = 720
    height = 460
    margin = 60
    plot_width = width - margin * 2
    plot_height = height - margin * 2

    def project_x(value: float) -> float:
        return margin + (value * plot_width)

    def project_y(value: float) -> float:
        return height - margin - (value * plot_height)

    def points_from_summary(summary: CalibrationSummary) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for bucket in summary.bins:
            if int(bucket["count"]) == 0:
                continue
            points.append((float(bucket["mean_prediction"]), float(bucket["observed_default_rate"])))
        return points

    def polyline(points: list[tuple[float, float]], color: str) -> str:
        if not points:
            return ""
        encoded_points = " ".join(f"{project_x(x):.1f},{project_y(y):.1f}" for x, y in points)
        circles = "\n".join(
            f'<circle cx="{project_x(x):.1f}" cy="{project_y(y):.1f}" r="4" fill="{color}" />'
            for x, y in points
        )
        return (
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{encoded_points}" />\n'
            f"{circles}"
        )

    raw_points = points_from_summary(raw_calibration)
    calibrated_points = points_from_summary(calibrated_calibration)
    diagonal = f"{project_x(0):.1f},{project_y(0):.1f} {project_x(1):.1f},{project_y(1):.1f}"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fffdf7" />
  <text x="{width / 2:.1f}" y="32" text-anchor="middle" font-size="22" font-family="Helvetica, Arial, sans-serif" fill="#1f2937">{title}</text>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#374151" stroke-width="2" />
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#374151" stroke-width="2" />
  <line x1="{project_x(0):.1f}" y1="{project_y(0):.1f}" x2="{project_x(1):.1f}" y2="{project_y(1):.1f}" stroke="#9ca3af" stroke-width="2" stroke-dasharray="8 6" />
  <text x="{width / 2:.1f}" y="{height - 16}" text-anchor="middle" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">Predicted default probability</text>
  <text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">Observed default rate</text>
  <text x="{width - 180}" y="58" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#b91c1c">Raw</text>
  <line x1="{width - 230}" y1="53" x2="{width - 190}" y2="53" stroke="#b91c1c" stroke-width="3" />
  <text x="{width - 180}" y="82" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#065f46">Calibrated</text>
  <line x1="{width - 230}" y1="77" x2="{width - 190}" y2="77" stroke="#065f46" stroke-width="3" />
  {polyline(raw_points, "#b91c1c")}
  {polyline(calibrated_points, "#065f46")}
</svg>
"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path


def write_calibration_report(output_path: str | Path, summary: CalibrationRunSummary) -> Path:
    """Write a markdown report describing the calibration effect."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validation = summary.splits["validation"]
    test = summary.splits["test"]
    lines = [
        "# Phase 5 Calibration Report",
        "",
        f"Run ID: `{summary.run_id}`",
        f"Selected candidate: `{summary.selected_candidate}`",
        f"Calibrator artifact: `{summary.calibrator_artifact_path}`",
        "",
        "## Reliability Curves",
        "",
        f"- Validation curve: `{summary.validation_curve_svg_path}`",
        f"- Test curve: `{summary.test_curve_svg_path}`",
        "",
        "## Before vs After",
        "",
        "| Split | Stage | AUC-ROC | AUC-PR | Brier | ECE | Max Gap | Rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for split_name, split_summary in (("validation", validation), ("test", test)):
        for stage_key, metric_key, calibration_key in (
            ("raw", "raw_metrics", "raw_calibration"),
            ("calibrated", "calibrated_metrics", "calibrated_calibration"),
        ):
            metrics = split_summary[metric_key]
            calibration = split_summary[calibration_key]
            lines.append(
                "| "
                f"{split_name} | {stage_key} | {metrics['auc_roc']:.6f} | {metrics['auc_pr']:.6f} | "
                f"{metrics['brier_score']:.6f} | {calibration['ece']:.6f} | "
                f"{calibration['max_calibration_gap']:.6f} | {metrics['row_count']} |"
            )

    lines.extend(
        [
            "",
            "## Calibration Interpretation",
            "",
            "- Validation calibration is fit using isotonic regression on the validation split probabilities.",
            "- Test metrics show how that calibration transfers to unseen examples from the held-out test split.",
            "- A well-calibrated model should keep reliability points close to the diagonal where predicted default probability matches observed default frequency.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for `python -m ml.models.calibrate`."""

    parser = argparse.ArgumentParser(description="Fit isotonic calibration for a trained model manifest.")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--experiment-dir", default="ml/models/experiments")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--modulo-sampling", type=int, default=1)
    parser.add_argument("--report-dir", default="ml/models/reports")
    parser.add_argument("--bins", type=int, default=DEFAULT_CALIBRATION_BINS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for calibration."""

    args = parse_args(argv)
    manifest_path = Path(args.manifest_path) if args.manifest_path else find_latest_manifest(args.experiment_dir)
    summary = calibrate_manifest(
        manifest_path,
        max_rows_per_split=args.max_rows_per_split,
        modulo_sampling=max(args.modulo_sampling, 1),
        report_dir=args.report_dir,
        bins=args.bins,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
