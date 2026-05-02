from datetime import UTC, datetime
from typing import Any

from models.audit_log import AuditLog


def build_explanation(audit_entries: list[AuditLog], decision_output: dict[str, Any]) -> dict[str, Any]:
    """
    Constructs a human-readable explanation from the raw audit trail.
    Templates handle successful, degraded, and manual review decisions.
    """
    decision = decision_output.get("decision")
    confidence = decision_output.get("confidence")
    rule_version = decision_output.get("rule_version")
    factors = _factor_objects(decision_output.get("factors", []))
    model_details = _model_details(audit_entries, decision_output)
    timeline = [_timeline_entry(entry) for entry in audit_entries]
    summary = _summary(decision, confidence, audit_entries, decision_output, model_details)

    return {
        "decision": decision,
        "summary": summary,
        "factors": factors,
        "model_factor_contributions": model_details["model_factor_contributions"],
        "timeline": timeline,
        "rule_version": rule_version,
        "model_version": model_details["model_version"],
        "generated_at": datetime.now(UTC),
    }


def _summary(
    decision: str | None,
    confidence: float | None,
    audit_entries: list[AuditLog],
    decision_output: dict[str, Any],
    model_details: dict[str, Any],
) -> str:
    degraded_steps = [
        entry
        for entry in audit_entries
        if entry.error_type is not None or entry.fallback_used
    ]
    drift_steps = [entry for entry in audit_entries if entry.step == "DRIFT_DETECTED"]
    threshold_reason = any(
        "Confidence below threshold" in str(factor)
        for factor in decision_output.get("factors", [])
    )

    if decision == "NEEDS_REVIEW" and threshold_reason:
        causes = _degradation_sentence(degraded_steps)
        confidence_text = _confidence_text(confidence)
        summary = (
            f"The system had insufficient reliable data to make an automatic decision. "
            f"{causes} Confidence {confidence_text} is below the required threshold, "
            "so this application has been sent for manual review."
        )
        return _append_warning_and_model_summary(summary, drift_steps, model_details)

    if degraded_steps:
        causes = _degradation_sentence(degraded_steps)
        summary = (
            f"Decision {decision} was produced with degraded data quality. "
            f"{causes} Confidence is {_confidence_text(confidence)}."
        )
        return _append_warning_and_model_summary(summary, drift_steps, model_details)

    summary = f"Decision {decision} was produced from verified data sources with confidence {_confidence_text(confidence)}."
    return _append_warning_and_model_summary(summary, drift_steps, model_details)


def _degradation_sentence(entries: list[AuditLog]) -> str:
    if not entries:
        return "No external data degradation was recorded."

    parts = []
    for entry in entries:
        label = entry.step.replace("_FETCH", "").replace("_", " ").title()
        status = entry.error_type or "fallback"
        parts.append(f"{label}: {status}")
    return "Data quality issues recorded: " + "; ".join(parts) + "."


def _factor_objects(factors: list[str]) -> list[dict[str, str]]:
    objects: list[dict[str, str]] = []
    for factor in factors:
        if " = " not in factor:
            continue
        name_source, value = factor.split(" = ", 1)
        if "(" in name_source and ")" in name_source:
            name = name_source.split("(", 1)[0].strip().replace("_", " ").title()
            status = name_source.split("(", 1)[1].split(")", 1)[0]
        else:
            name = name_source.replace("_", " ").title()
            status = "derived"
        objects.append({"name": name, "value": value, "status": status})
    return objects


def _timeline_entry(entry: AuditLog) -> dict[str, Any]:
    status = entry.error_type
    if status is None and entry.output_snapshot:
        status = str(entry.output_snapshot.get("decision") or entry.output_snapshot.get("status") or "SUCCESS")
    return {
        "step": entry.step,
        "status": status or "SUCCESS",
        "timestamp": entry.created_at,
    }


def _confidence_text(confidence: float | None) -> str:
    return "unknown" if confidence is None else f"{confidence:.2f}"


def _model_details(audit_entries: list[AuditLog], decision_output: dict[str, Any]) -> dict[str, Any]:
    payload = _model_payload_from_decision(decision_output)
    if payload["model_factor_contributions"] or payload["model_version"]:
        return payload

    for entry in reversed(audit_entries):
        if entry.step == "ML_SCORING" and entry.output_snapshot:
            return _model_payload_from_decision(entry.output_snapshot)

    return {
        "model_factor_contributions": [],
        "model_summary": None,
        "model_version": None,
    }


def _model_payload_from_decision(payload: dict[str, Any]) -> dict[str, Any]:
    contributions = []
    for row in payload.get("model_factor_contributions", []):
        if not isinstance(row, dict):
            continue
        feature_name = str(row.get("feature_name") or "").strip()
        if not feature_name:
            continue
        shap_value = float(row.get("shap_contribution", 0.0))
        direction = str(row.get("direction") or ("increase_default_risk" if shap_value >= 0 else "decrease_default_risk"))
        contributions.append(
            {
                "feature_name": feature_name,
                "raw_value": str(row.get("raw_value", "unknown")),
                "shap_contribution": round(shap_value, 6),
                "direction": direction,
            }
        )

    return {
        "model_factor_contributions": contributions,
        "model_summary": payload.get("model_summary") or _model_summary_from_contributions(contributions),
        "model_version": payload.get("model_version"),
    }


def _append_model_summary(summary: str, model_details: dict[str, Any]) -> str:
    model_summary = model_details.get("model_summary")
    if not model_summary:
        return summary
    return f"{summary} {model_summary}"


def _append_warning_and_model_summary(
    summary: str,
    drift_steps: list[AuditLog],
    model_details: dict[str, Any],
) -> str:
    summary_with_warning = _append_drift_summary(summary, drift_steps)
    return _append_model_summary(summary_with_warning, model_details)


def _append_drift_summary(summary: str, drift_steps: list[AuditLog]) -> str:
    if not drift_steps:
        return summary
    latest = drift_steps[-1]
    drifted_features = [
        str(row.get("feature_name"))
        for row in (latest.output_snapshot or {}).get("drifted_features", [])
        if row.get("feature_name")
    ]
    if drifted_features:
        featured_text = ", ".join(drifted_features[:3])
        return f"{summary} An ML drift warning was recorded for {featured_text}."
    return f"{summary} An ML drift warning was recorded during scoring."


def _model_summary_from_contributions(contributions: list[dict[str, Any]]) -> str | None:
    if not contributions:
        return None

    ranked = sorted(contributions, key=lambda row: -abs(float(row["shap_contribution"])))
    increases = [row for row in ranked if row["direction"] == "increase_default_risk"][:2]
    decreases = [row for row in ranked if row["direction"] == "decrease_default_risk"][:1]

    fragments: list[str] = []
    if increases:
        fragments.append(
            _serialise_model_fragment(
                increases,
                "increased predicted default risk",
            )
        )
    if decreases:
        fragments.append(
            _serialise_model_fragment(
                decreases,
                "reduced predicted default risk",
            )
        )

    if not fragments:
        return None
    if len(fragments) == 1:
        return f"Model factors: {fragments[0]}."
    return f"Model factors: {fragments[0]}, while {fragments[1]}."


def _serialise_model_fragment(rows: list[dict[str, Any]], trailing_text: str) -> str:
    phrases = [f"{row['feature_name']} ({row['raw_value']})" for row in rows]
    if len(phrases) == 1:
        return f"{phrases[0]} {trailing_text}"
    return f"{', '.join(phrases[:-1])} and {phrases[-1]} {trailing_text}"
