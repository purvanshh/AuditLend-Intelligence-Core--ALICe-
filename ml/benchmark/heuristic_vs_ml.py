"""Head-to-head benchmark for heuristic versus deployed ML scoring."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from engine.confidence import compute_decision_confidence
from engine.rule_sets import RULE_SET_V1, RULE_SET_V2
from engine.rules import Decision, evaluate
from engine.scoring import compute_risk_score
from ml.governance.ab_test import ABTestReport, OutcomeRecord, summarize_outcomes
from ml.models.evaluate import find_latest_manifest, load_manifest, load_model_artifact
from ml.models.train import predict_probabilities, prepare_training_dataset


@dataclass(frozen=True)
class BenchmarkReport:
    """Serializable benchmark output comparing heuristic and deployed ML."""

    run_id: str
    manifest_path: str
    selected_candidate: str
    row_count: int
    confidence_threshold: float
    ab_report: dict[str, Any]
    report_path: str


def benchmark_manifest(
    manifest_path: str | Path,
    *,
    confidence_threshold: float = 0.6,
    max_rows_per_split: int | None = None,
    modulo_sampling: int = 1,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    report_dir: str | Path = "ml/benchmark/reports",
) -> BenchmarkReport:
    """Benchmark heuristic control against deployed ML behavior on the test split."""

    manifest = load_manifest(manifest_path)
    prepared_dataset = prepare_training_dataset(
        max_rows_per_split=max_rows_per_split,
        modulo_sampling=modulo_sampling,
        env_var=env_var,
    )
    test_split = prepared_dataset.split_matrices["test"]
    if test_split.row_count == 0:
        raise RuntimeError("The test split is empty; benchmark cannot run.")

    model = load_model_artifact(manifest["artifact_path"])
    raw_probabilities = predict_probabilities(model, test_split.X, feature_names=test_split.feature_names)
    calibrated_probabilities = _apply_optional_calibrator(
        Path(manifest["artifact_path"]).parent / "isotonic_calibrator.pkl",
        raw_probabilities,
    )

    rows: list[OutcomeRecord] = []
    for feature_row, raw_probability, calibrated_probability in zip(
        test_split.rows,
        raw_probabilities,
        calibrated_probabilities,
        strict=True,
    ):
        heuristic_outcome = _heuristic_outcome(feature_row)
        ml_outcome = _ml_outcome(
            feature_row,
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            confidence_threshold=confidence_threshold,
            fallback_outcome=heuristic_outcome,
        )
        rows.append(heuristic_outcome)
        rows.append(ml_outcome)

    ab_report = summarize_outcomes(rows, ml_ratio=0.5)
    report_dir_path = Path(report_dir)
    report_dir_path.mkdir(parents=True, exist_ok=True)
    report_path = report_dir_path / f"{manifest['run_id']}_heuristic_vs_ml.md"
    write_benchmark_report(report_path, manifest, ab_report, confidence_threshold=confidence_threshold)
    return BenchmarkReport(
        run_id=str(manifest["run_id"]),
        manifest_path=str(Path(manifest_path)),
        selected_candidate=str(manifest["selected_candidate"]),
        row_count=test_split.row_count,
        confidence_threshold=confidence_threshold,
        ab_report=ab_report.to_dict(),
        report_path=str(report_path),
    )


def write_benchmark_report(
    report_path: str | Path,
    manifest: dict[str, Any],
    ab_report: ABTestReport,
    *,
    confidence_threshold: float,
) -> Path:
    """Write a markdown report summarizing benchmark outcomes."""

    heuristic = next(row for row in ab_report.arms if row.arm == "heuristic")
    ml = next(row for row in ab_report.arms if row.arm == "ml")

    lines = [
        "# Phase 9 Heuristic vs ML Benchmark",
        "",
        f"Run ID: `{manifest['run_id']}`",
        f"Selected candidate: `{manifest['selected_candidate']}`",
        f"Confidence threshold: `{confidence_threshold:.2f}`",
        "",
        "## Assumptions",
        "",
        "- Heuristic benchmark uses a deterministic income-stability proxy derived from the engineered feature set.",
        "- ML benchmark uses calibrated probabilities when available and falls back to the heuristic decision when model confidence is below threshold.",
        "- Simulated profit uses `+12%` of loan amount for performing approved loans and `-65%` loss given default for approved loans that default.",
        "",
        "## Arm Comparison",
        "",
        "| Arm | Rows | Approval Rate | Decline Rate | Review Rate | Avg Confidence | Default Rate on Approved | Simulated Profit | Profit / App |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in (heuristic, ml):
        lines.append(
            f"| {summary.arm} | {summary.row_count} | {summary.approval_rate:.4f} | "
            f"{summary.decline_rate:.4f} | {summary.manual_review_rate:.4f} | "
            f"{summary.average_confidence:.4f} | {summary.default_rate_on_approved:.4f} | "
            f"{summary.simulated_profit:.2f} | {summary.simulated_profit_per_application:.2f} |"
        )

    lines.extend(
        [
            "",
            "## ML Minus Heuristic",
            "",
            f"- Approval rate delta: `{ab_report.approval_rate_delta_ml_minus_heuristic:.4f}`",
            f"- Default rate delta on approved loans: `{ab_report.default_rate_delta_ml_minus_heuristic:.4f}`",
            f"- Simulated profit delta: `{ab_report.profit_delta_ml_minus_heuristic:.2f}`",
        ]
    )

    path = Path(report_path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark heuristic versus deployed ML scoring.")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--modulo-sampling", type=int, default=1)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest_path) if args.manifest_path else find_latest_manifest()
    report = benchmark_manifest(
        manifest_path,
        confidence_threshold=float(args.confidence_threshold),
        max_rows_per_split=args.max_rows_per_split,
        modulo_sampling=max(int(args.modulo_sampling), 1),
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


def _heuristic_outcome(feature_row: dict[str, Any]) -> OutcomeRecord:
    credit_score = int(round(float(feature_row.get("credit_score_midpoint", 600.0))))
    income_stability = _proxy_income_stability(feature_row)
    dti = float(feature_row.get("dti_ratio", 0.0))
    gst_compliant = bool(feature_row.get("tax_lien_flag", 0.0) == 0.0 and feature_row.get("bankruptcy_flag", 0.0) == 0.0)

    risk_score, _ = compute_risk_score(credit_score, income_stability, dti, gst_compliant, [], RULE_SET_V1)
    decision, _ = evaluate(risk_score, credit_score, dti, [], gst_compliant, RULE_SET_V1)
    confidence, _ = compute_decision_confidence(risk_score, decision, 1.0, [])
    if confidence < 0.6:
        decision = Decision.NEEDS_REVIEW

    return OutcomeRecord(
        arm="heuristic",
        decision=decision.value,
        confidence=float(confidence),
        defaulted=int(feature_row.get("target_defaulted", 0)),
        loan_amount=float(feature_row.get("loan_amount", 0.0)),
        scoring_strategy="heuristic",
    )


def _ml_outcome(
    feature_row: dict[str, Any],
    *,
    raw_probability: float,
    calibrated_probability: float,
    confidence_threshold: float,
    fallback_outcome: OutcomeRecord,
) -> OutcomeRecord:
    model_confidence = max(calibrated_probability, 1.0 - calibrated_probability)
    if model_confidence < confidence_threshold:
        return OutcomeRecord(
            arm="ml",
            decision=fallback_outcome.decision,
            confidence=fallback_outcome.confidence,
            defaulted=fallback_outcome.defaulted,
            loan_amount=fallback_outcome.loan_amount,
            scoring_strategy="heuristic_fallback",
        )

    risk_score = round((1.0 - calibrated_probability) * 100.0, 2)
    credit_score = int(round(float(feature_row.get("credit_score_midpoint", 600.0))))
    dti = float(feature_row.get("dti_ratio", 0.0))
    gst_compliant = bool(feature_row.get("tax_lien_flag", 0.0) == 0.0 and feature_row.get("bankruptcy_flag", 0.0) == 0.0)

    decision, _ = evaluate(risk_score, credit_score, dti, [], gst_compliant, RULE_SET_V2)
    confidence, _ = compute_decision_confidence(risk_score, decision, 1.0, [])
    if confidence < confidence_threshold:
        decision = Decision.NEEDS_REVIEW

    return OutcomeRecord(
        arm="ml",
        decision=decision.value,
        confidence=float(confidence),
        defaulted=int(feature_row.get("target_defaulted", 0)),
        loan_amount=float(feature_row.get("loan_amount", 0.0)),
        scoring_strategy="ml",
    )


def _proxy_income_stability(feature_row: dict[str, Any]) -> float:
    headroom = float(feature_row.get("credit_card_headroom_ratio", 0.5))
    clean_history = float(feature_row.get("never_delinquent_ratio", 0.5))
    emi_burden = float(feature_row.get("existing_emi_to_income", 0.0))
    inquiry_pressure = float(feature_row.get("recent_inquiry_pressure", 0.0))
    proxy = (headroom * 0.4) + (clean_history * 0.4) + ((1.0 - min(emi_burden, 1.0)) * 0.15) + ((1.0 - min(inquiry_pressure, 1.0)) * 0.05)
    return min(max(proxy, 0.0), 1.0)


def _apply_optional_calibrator(calibrator_path: Path, raw_probabilities: Sequence[float]) -> list[float]:
    if not calibrator_path.exists():
        return [float(value) for value in raw_probabilities]
    with calibrator_path.open("rb") as handle:
        calibrator = pickle.load(handle)
    calibrated = calibrator.predict(list(raw_probabilities))
    return [min(max(float(value), 0.0), 1.0) for value in calibrated]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
