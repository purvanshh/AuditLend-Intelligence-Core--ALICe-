"""SHAP-based per-prediction explanations for AuditLend ML scoring."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ml.models.evaluate import load_manifest, load_model_artifact
from ml.models.train import MODEL_CATEGORICAL_FEATURES, encode_feature_row, predict_probabilities


DISPLAY_NAME_OVERRIDES: dict[str, str] = {
    "bc_util_ratio": "Bankcard Utilization",
    "credit_score_midpoint": "Credit Score",
    "dti_ratio": "Debt-To-Income Ratio",
    "loan_amount_to_income": "Loan-To-Income Ratio",
    "revol_util_ratio": "Revolving Utilization",
    "verification_status": "Verification Status",
}
PERCENT_LIKE_FEATURES = {
    "all_util_ratio",
    "bc_util_ratio",
    "credit_card_headroom_ratio",
    "dti_ratio",
    "existing_emi_to_income",
    "high_utilization_fraction",
    "il_util_ratio",
    "installment_to_income",
    "loan_amount_to_income",
    "mortgage_account_share",
    "never_delinquent_ratio",
    "open_account_density",
    "revol_util_ratio",
    "total_bc_limit_to_income",
    "total_balance_to_income",
    "total_rev_limit_to_income",
}
INCOME_BANDED_FEATURES = {"monthly_income"}
AMOUNT_BANDED_FEATURES = {
    "balance_per_open_account",
    "estimated_existing_emi",
    "funded_amount",
    "installment",
    "loan_amount",
}


@dataclass(frozen=True)
class ModelFactorContribution:
    """One explainable model factor for an individual prediction."""

    feature_name: str
    raw_value: str
    shap_contribution: float
    direction: str


@dataclass(frozen=True)
class PredictionExplanation:
    """Serializable per-prediction explanation payload."""

    model_version: str
    selected_candidate: str
    predicted_default_probability: float
    calibrated_default_probability: float | None
    explanation_method: str
    baseline_value: float
    model_factor_contributions: list[ModelFactorContribution]
    model_summary: str

    def to_audit_payload(self) -> dict[str, Any]:
        """Render an audit-log-friendly explanation snapshot."""

        payload = asdict(self)
        payload["model_factor_contributions"] = [asdict(row) for row in self.model_factor_contributions]
        return json.loads(json.dumps(payload))


def explain_feature_row(
    feature_row: dict[str, Any],
    manifest_path: str | Path,
    *,
    max_features: int = 8,
) -> PredictionExplanation:
    """Explain one engineered feature row using the trained Phase 3/5 artifact bundle."""

    manifest = load_manifest(manifest_path)
    model = load_model_artifact(manifest["artifact_path"])
    feature_names = [str(name) for name in manifest["feature_names"]]
    categories_by_feature = {
        str(feature): [str(category) for category in categories]
        for feature, categories in manifest.get("categories_by_feature", {}).items()
    }

    encoded_row = encode_feature_row(feature_row, categories_by_feature)
    model_input = _build_model_input(encoded_row, feature_names)
    predicted_default_probability = predict_probabilities(model, [encoded_row], feature_names=feature_names)[0]
    calibrated_default_probability = _apply_optional_calibrator(
        Path(manifest["artifact_path"]).parent / "isotonic_calibrator.pkl",
        predicted_default_probability,
    )

    shap_values, baseline_value = _compute_shap_values(model, model_input)
    contributions = _aggregate_contributions(
        feature_row,
        feature_names,
        shap_values,
        max_features=max_features,
    )
    model_summary = _build_model_summary(contributions)

    return PredictionExplanation(
        model_version=str(manifest["run_id"]),
        selected_candidate=str(manifest["selected_candidate"]),
        predicted_default_probability=round(float(predicted_default_probability), 6),
        calibrated_default_probability=(
            round(float(calibrated_default_probability), 6)
            if calibrated_default_probability is not None
            else None
        ),
        explanation_method="shap",
        baseline_value=round(float(baseline_value), 6),
        model_factor_contributions=contributions,
        model_summary=model_summary,
    )


def _build_model_input(encoded_row: Sequence[float], feature_names: Sequence[str]):
    try:
        import pandas as pd
    except ImportError:
        return [list(encoded_row)]
    return pd.DataFrame([list(encoded_row)], columns=list(feature_names))


def _compute_shap_values(model: Any, model_input: Any) -> tuple[list[float], float]:
    import numpy as np
    import shap

    if _supports_tree_explainer(model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(model_input, check_additivity=False)
        baseline_value = explainer.expected_value
    else:
        explainer = shap.Explainer(model)
        explanation = explainer(model_input)
        shap_values = explanation.values
        baseline_value = explanation.base_values

    return _normalize_shap_output(shap_values, baseline_value, np)


def _supports_tree_explainer(model: Any) -> bool:
    model_type = type(model).__name__.lower()
    module_name = type(model).__module__.lower()
    return hasattr(model, "feature_importances_") or "lightgbm" in module_name or "xgboost" in module_name or (
        "tree" in model_type or "forest" in model_type
    )


def _normalize_shap_output(shap_values: Any, baseline_value: Any, np_module: Any) -> tuple[list[float], float]:
    if isinstance(shap_values, list):
        selected_values = shap_values[-1]
    else:
        selected_values = shap_values

    values = np_module.asarray(selected_values, dtype=float)
    if values.ndim == 3:
        normalized_values = values[0, :, -1]
    elif values.ndim == 2:
        normalized_values = values[0]
    elif values.ndim == 1:
        normalized_values = values
    else:
        raise ValueError("Unsupported SHAP value shape for prediction explanation.")

    base_values = np_module.asarray(baseline_value, dtype=float)
    if base_values.ndim == 0:
        normalized_base = float(base_values)
    elif base_values.ndim == 1:
        normalized_base = float(base_values[-1])
    else:
        normalized_base = float(base_values.reshape(-1)[-1])

    return [float(value) for value in normalized_values.tolist()], normalized_base


def _aggregate_contributions(
    feature_row: dict[str, Any],
    feature_names: Sequence[str],
    shap_values: Sequence[float],
    *,
    max_features: int,
) -> list[ModelFactorContribution]:
    grouped: dict[str, dict[str, Any]] = {}

    for feature_name, shap_value in zip(feature_names, shap_values):
        group_key = _group_key(feature_name)
        group = grouped.setdefault(
            group_key,
            {
                "feature_name": _display_name(group_key),
                "raw_value": _format_feature_value(group_key, feature_row.get(group_key)),
                "shap_contribution": 0.0,
            },
        )
        group["shap_contribution"] += float(shap_value)

    ranked_rows = sorted(
        grouped.values(),
        key=lambda row: (-abs(float(row["shap_contribution"])), str(row["feature_name"])),
    )[:max(max_features, 1)]

    return [
        ModelFactorContribution(
            feature_name=str(row["feature_name"]),
            raw_value=str(row["raw_value"]),
            shap_contribution=round(float(row["shap_contribution"]), 6),
            direction="increase_default_risk" if float(row["shap_contribution"]) >= 0 else "decrease_default_risk",
        )
        for row in ranked_rows
    ]


def _group_key(feature_name: str) -> str:
    if "=" in feature_name:
        candidate = feature_name.split("=", 1)[0]
        if candidate in MODEL_CATEGORICAL_FEATURES:
            return candidate
    return feature_name


def _display_name(feature_name: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(feature_name, feature_name.replace("_", " ").title())


def _format_feature_value(feature_name: str, value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if feature_name in INCOME_BANDED_FEATURES:
        return _income_band(value)
    if feature_name in AMOUNT_BANDED_FEATURES:
        return _amount_band(value)
    if feature_name in PERCENT_LIKE_FEATURES:
        return f"{float(value) * 100:.1f}%"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:.2f}"


def _build_model_summary(contributions: Sequence[ModelFactorContribution]) -> str:
    if not contributions:
        return "Model factors were unavailable for this decision."

    increases = [row for row in contributions if row.direction == "increase_default_risk"][:2]
    decreases = [row for row in contributions if row.direction == "decrease_default_risk"][:1]

    fragments: list[str] = []
    if increases:
        fragments.append(
            _join_phrases(
                [
                    f"{row.feature_name} ({row.raw_value}) increased predicted default risk"
                    for row in increases
                ]
            )
        )
    if decreases:
        fragments.append(
            _join_phrases(
                [
                    f"{row.feature_name} ({row.raw_value}) reduced predicted default risk"
                    for row in decreases
                ]
            )
        )

    if not fragments:
        top_row = contributions[0]
        return (
            f"Model factors: {top_row.feature_name} ({top_row.raw_value}) "
            "was the strongest signal for this prediction."
        )

    if len(fragments) == 1:
        return f"Model factors: {fragments[0]}."
    return f"Model factors: {fragments[0]}, while {fragments[1]}."


def _join_phrases(phrases: Sequence[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + f" and {phrases[-1]}"


def _apply_optional_calibrator(calibrator_path: Path, probability: float) -> float | None:
    if not calibrator_path.exists():
        return None
    with calibrator_path.open("rb") as handle:
        calibrator = pickle.load(handle)
    calibrated = calibrator.predict([float(probability)])
    return min(max(float(calibrated[0]), 0.0), 1.0)


def _income_band(value: Any) -> str:
    income = float(value)
    if income <= 0:
        return "UNKNOWN"
    if income < 25_000:
        return "0-25K"
    if income < 50_000:
        return "25K-50K"
    if income < 100_000:
        return "50K-1L"
    if income < 200_000:
        return "1L-2L"
    return "2L+"


def _amount_band(value: Any) -> str:
    amount = float(value)
    if amount <= 0:
        return "UNKNOWN"
    if amount < 100_000:
        return "0-1L"
    if amount < 500_000:
        return "1L-5L"
    if amount < 1_000_000:
        return "5L-10L"
    return "10L+"
