from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from ml.governance.drift_detector import detect_feature_drift
from services.metrics import drift_alerts_evidently_total, drift_reports_generated_total

try:
    from evidently.report import Report
    from evidently.metrics import DataDriftTable
    from evidently.test_suite import TestSuite
    from evidently.tests import TestFeatureValueDrift
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

logger = structlog.get_logger()


class EvidentlyDriftReporter:
    def __init__(self, reference_data: pd.DataFrame) -> None:
        self.reference_data = reference_data

    def compute_data_drift(
        self,
        candidate_data: pd.DataFrame,
        drift_share: float = 0.1,
    ) -> dict[str, Any]:
        if EVIDENTLY_AVAILABLE:
            return self._compute_with_evidently(candidate_data, drift_share)
        return self._compute_with_ks_fallback(candidate_data, drift_share)

    def generate_html_report(
        self,
        candidate_data: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if EVIDENTLY_AVAILABLE:
            report = Report(metrics=[DataDriftTable()])
            report.run(reference_data=self.reference_data, current_data=candidate_data)
            report.save_html(str(output))
            drift_reports_generated_total.labels(format="html").inc()
        else:
            result = self.compute_data_drift(candidate_data)
            output.write_text(json.dumps(result, indent=2))
            drift_reports_generated_total.labels(format="json").inc()

        logger.info("drift_report_generated", path=str(output))
        return output

    def generate_test_suite(
        self,
        candidate_data: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if EVIDENTLY_AVAILABLE:
            common_cols = list(set(self.reference_data.columns) & set(candidate_data.columns))
            tests = [TestFeatureValueDrift(column_name=col) for col in common_cols]
            suite = TestSuite(tests=tests)
            suite.run(reference_data=self.reference_data, current_data=candidate_data)
            suite.save_html(str(output))
        else:
            result = self.compute_data_drift(candidate_data)
            output.write_text(json.dumps(result, indent=2))

        logger.info("drift_test_suite_generated", path=str(output))
        return output

    def _compute_with_evidently(
        self,
        candidate_data: pd.DataFrame,
        threshold: float,
    ) -> dict[str, Any]:
        report = Report(metrics=[DataDriftTable()])
        report.run(reference_data=self.reference_data, current_data=candidate_data)
        report_dict = report.as_dict()

        metrics_list = report_dict.get("metrics", [])
        drift_result = {}
        for metric_entry in metrics_list:
            result = metric_entry.get("result", {})
            if "drift_by_columns" in result:
                drift_result = result
                break

        drift_by_columns = drift_result.get("drift_by_columns", {})
        feature_drift_scores: dict[str, float] = {}
        drifted_features: list[str] = []

        for col_name, col_data in drift_by_columns.items():
            score = float(col_data.get("drift_score", 0.0))
            feature_drift_scores[col_name] = score
            if col_data.get("drift_detected", False):
                drifted_features.append(col_name)
                drift_alerts_evidently_total.labels(feature=col_name).inc()

        total = len(feature_drift_scores)
        drift_share_val = len(drifted_features) / total if total > 0 else 0.0

        logger.info(
            "evidently_drift_computed",
            total_features=total,
            drifted=len(drifted_features),
            drift_share=round(drift_share_val, 4),
        )

        return {
            "drift_share": drift_share_val,
            "drifted_features": drifted_features,
            "feature_drift_scores": feature_drift_scores,
            "dataset_drift": drift_share_val >= threshold,
        }

    def _compute_with_ks_fallback(
        self,
        candidate_data: pd.DataFrame,
        threshold: float,
    ) -> dict[str, Any]:
        ref_rows = self.reference_data.to_dict(orient="records")
        cand_rows = candidate_data.to_dict(orient="records")

        drift_report = detect_feature_drift(
            ref_rows,
            cand_rows,
            increment_metrics=True,
        )

        feature_drift_scores: dict[str, float] = {}
        drifted_features: list[str] = []
        for fdr in drift_report.checked_features:
            feature_drift_scores[fdr.feature_name] = fdr.ks_statistic
            if fdr.alert_triggered:
                drifted_features.append(fdr.feature_name)

        total = drift_report.total_features
        drift_share_val = len(drifted_features) / total if total > 0 else 0.0

        return {
            "drift_share": drift_share_val,
            "drifted_features": drifted_features,
            "feature_drift_scores": feature_drift_scores,
            "dataset_drift": drift_share_val >= threshold,
        }


def create_monitoring_dashboard(
    reference_data: pd.DataFrame,
    candidate_data: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reporter = EvidentlyDriftReporter(reference_data)

    report_file = "drift_report.html" if EVIDENTLY_AVAILABLE else "drift_report.json"
    report_path = output_path / report_file
    reporter.generate_html_report(candidate_data, report_path)

    suite_file = "test_suite.html" if EVIDENTLY_AVAILABLE else "test_suite.json"
    suite_path = output_path / suite_file
    reporter.generate_test_suite(candidate_data, suite_path)

    return [report_path, suite_path]
