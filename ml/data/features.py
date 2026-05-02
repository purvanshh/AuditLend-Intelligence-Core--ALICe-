"""Feature engineering and exploratory summaries for the AuditLend ML pipeline."""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

FEATURE_COLUMNS: tuple[str, ...] = (
    "loan_amount",
    "funded_amount",
    "term_months",
    "interest_rate_pct",
    "installment",
    "monthly_income",
    "estimated_existing_emi",
    "dti_ratio",
    "loan_amount_to_income",
    "installment_to_income",
    "existing_emi_to_income",
    "credit_score_midpoint",
    "credit_score_recent_delta",
    "credit_history_age_years",
    "employment_length_years",
    "revol_util_ratio",
    "bc_util_ratio",
    "all_util_ratio",
    "il_util_ratio",
    "revol_balance_to_income",
    "current_balance_to_income",
    "total_balance_to_income",
    "total_rev_limit_to_income",
    "total_bc_limit_to_income",
    "credit_card_headroom_ratio",
    "delinquency_burden",
    "recent_inquiry_pressure",
    "credit_inquiry_velocity",
    "open_account_density",
    "accounts_per_year",
    "balance_per_open_account",
    "mortgage_account_share",
    "bankruptcy_flag",
    "tax_lien_flag",
    "high_utilization_fraction",
    "never_delinquent_ratio",
    "collections_12m",
    "recent_revolving_trade_gap_months",
    "revolving_trade_age_years",
    "open_revolving_24m",
    "open_installment_24m",
    "target_defaulted",
)

CORRELATION_REPORT_COLUMNS: tuple[str, ...] = (
    "loan_amount",
    "monthly_income",
    "estimated_existing_emi",
    "dti_ratio",
    "loan_amount_to_income",
    "installment_to_income",
    "credit_score_midpoint",
    "credit_history_age_years",
    "revol_util_ratio",
    "bc_util_ratio",
    "delinquency_burden",
    "recent_inquiry_pressure",
)


def build_feature_row(clean_row: dict[str, Any]) -> dict[str, Any]:
    """Convert one cleaned row into model-ready numeric and categorical features."""

    monthly_income = _float_or_zero(clean_row.get("monthly_income"))
    loan_amount = _float_or_zero(clean_row.get("loan_amount"))
    installment = _float_or_zero(clean_row.get("installment"))
    estimated_existing_emi = _float_or_zero(clean_row.get("estimated_existing_emi"))
    credit_history_age_years = _compute_credit_history_age_years(clean_row)
    total_acc = _float_or_zero(clean_row.get("total_acc"))
    open_acc = _float_or_zero(clean_row.get("open_acc"))
    fico_midpoint = clean_row.get("fico_midpoint")
    last_fico_midpoint = clean_row.get("last_fico_midpoint")
    total_bc_limit = _float_or_zero(clean_row.get("total_bc_limit"))
    bc_util_ratio = _safe_ratio(clean_row.get("bc_util_pct"), 100.0)

    return {
        "loan_id": clean_row.get("loan_id"),
        "issue_date": clean_row.get("issue_date"),
        "loan_status": clean_row.get("loan_status"),
        "grade": clean_row.get("grade"),
        "sub_grade": clean_row.get("sub_grade"),
        "purpose": clean_row.get("purpose"),
        "home_ownership": clean_row.get("home_ownership"),
        "verification_status": clean_row.get("verification_status"),
        "loan_amount": loan_amount,
        "funded_amount": _float_or_zero(clean_row.get("funded_amount")),
        "term_months": _float_or_zero(clean_row.get("term_months")),
        "interest_rate_pct": _float_or_zero(clean_row.get("interest_rate_pct")),
        "installment": installment,
        "monthly_income": monthly_income,
        "estimated_existing_emi": estimated_existing_emi,
        "dti_ratio": _safe_ratio(clean_row.get("dti_pct"), 100.0),
        "loan_amount_to_income": _safe_ratio(loan_amount, monthly_income),
        "installment_to_income": _safe_ratio(installment, monthly_income),
        "existing_emi_to_income": _safe_ratio(estimated_existing_emi, monthly_income),
        "credit_score_midpoint": _float_or_zero(fico_midpoint),
        "credit_score_recent_delta": _safe_delta(last_fico_midpoint, fico_midpoint),
        "credit_history_age_years": credit_history_age_years,
        "employment_length_years": _float_or_zero(clean_row.get("employment_length_years")),
        "revol_util_ratio": _safe_ratio(clean_row.get("revol_util_pct"), 100.0),
        "bc_util_ratio": bc_util_ratio,
        "all_util_ratio": _safe_ratio(clean_row.get("all_util_pct"), 100.0),
        "il_util_ratio": _safe_ratio(clean_row.get("il_util_pct"), 100.0),
        "revol_balance_to_income": _safe_ratio(clean_row.get("revol_bal"), monthly_income),
        "current_balance_to_income": _safe_ratio(clean_row.get("tot_cur_bal"), monthly_income),
        "total_balance_to_income": _safe_ratio(clean_row.get("total_bal_ex_mort"), monthly_income),
        "total_rev_limit_to_income": _safe_ratio(clean_row.get("total_rev_hi_lim"), monthly_income),
        "total_bc_limit_to_income": _safe_ratio(total_bc_limit, monthly_income),
        "credit_card_headroom_ratio": max(0.0, 1.0 - bc_util_ratio),
        "delinquency_burden": _safe_ratio(clean_row.get("delinq_2yrs"), total_acc),
        "recent_inquiry_pressure": _safe_ratio(clean_row.get("inq_last_6mths"), 6.0),
        "credit_inquiry_velocity": _safe_ratio(clean_row.get("inq_last_12m"), 12.0),
        "open_account_density": _safe_ratio(open_acc, total_acc),
        "accounts_per_year": _safe_ratio(total_acc, credit_history_age_years),
        "balance_per_open_account": _safe_ratio(clean_row.get("tot_cur_bal"), open_acc),
        "mortgage_account_share": _safe_ratio(clean_row.get("mort_acc"), total_acc),
        "bankruptcy_flag": 1.0 if _float_or_zero(clean_row.get("pub_rec_bankruptcies")) > 0 else 0.0,
        "tax_lien_flag": 1.0 if _float_or_zero(clean_row.get("tax_liens")) > 0 else 0.0,
        "high_utilization_fraction": _safe_ratio(clean_row.get("percent_bc_gt_75"), 100.0),
        "never_delinquent_ratio": _safe_ratio(clean_row.get("pct_tl_nvr_dlq"), 100.0),
        "collections_12m": _float_or_zero(clean_row.get("collections_12_mths_ex_med")),
        "recent_revolving_trade_gap_months": _float_or_zero(clean_row.get("mo_sin_rcnt_rev_tl_op")),
        "revolving_trade_age_years": _safe_ratio(clean_row.get("mo_sin_old_rev_tl_op"), 12.0),
        "open_revolving_24m": _float_or_zero(clean_row.get("open_rv_24m")),
        "open_installment_24m": _float_or_zero(clean_row.get("open_il_24m")),
        "target_defaulted": float(clean_row.get("defaulted", 0)),
    }


def build_feature_frame(clean_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build an in-memory feature frame from cleaned rows."""

    return [build_feature_row(row) for row in clean_rows]


def iter_feature_rows(clean_rows: Iterable[dict[str, Any]]):
    """Yield feature rows lazily for large-corpus workflows."""

    for row in clean_rows:
        yield build_feature_row(row)


def compute_correlation_matrix(
    feature_rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute a deterministic Pearson correlation matrix."""

    columns = tuple(feature_names or CORRELATION_REPORT_COLUMNS)
    matrix: dict[str, dict[str, float]] = {}
    for left in columns:
        matrix[left] = {}
        left_values = [_float_or_zero(row.get(left)) for row in feature_rows]
        for right in columns:
            right_values = [_float_or_zero(row.get(right)) for row in feature_rows]
            matrix[left][right] = _pearson(left_values, right_values)
    return matrix


def render_correlation_heatmap(
    correlation_matrix: dict[str, dict[str, float]],
    *,
    feature_names: Sequence[str] | None = None,
) -> str:
    """Render a markdown heatmap with correlation bands."""

    columns = tuple(feature_names or correlation_matrix.keys())
    lines = [
        "| feature | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join(["---:"] * len(columns)) + " |",
    ]
    for left in columns:
        row_values = []
        for right in columns:
            value = correlation_matrix[left][right]
            row_values.append(f"{value:.2f} {_band_label(value)}")
        lines.append("| " + left + " | " + " | ".join(row_values) + " |")
    return "\n".join(lines)


def write_feature_correlation_report(
    output_path: str | Path,
    *,
    feature_rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str] | None = None,
) -> Path:
    """Write a markdown report with a textual correlation heatmap."""

    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(feature_names or CORRELATION_REPORT_COLUMNS)
    matrix = compute_correlation_matrix(feature_rows, columns)
    report_lines = [
        "# Phase 2 Feature Correlation Heatmap",
        "",
        "Generated on 2026-05-02 from the Phase 2 engineered feature set.",
        "",
        "Legend:",
        "",
        "- `++` strong positive correlation (`>= 0.70`)",
        "- `+` moderate positive correlation (`0.30` to `0.69`)",
        "- `.` weak correlation (`-0.29` to `0.29`)",
        "- `-` moderate negative correlation (`-0.30` to `-0.69`)",
        "- `--` strong negative correlation (`<= -0.70`)",
        "",
        render_correlation_heatmap(matrix, feature_names=columns),
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report_path


def sample_feature_rows(
    feature_rows: Iterable[dict[str, Any]],
    *,
    modulo: int = 97,
    limit: int = 25000,
) -> list[dict[str, Any]]:
    """Select a deterministic subset for exploratory reporting."""

    sampled_rows: list[dict[str, Any]] = []
    for row in feature_rows:
        loan_id = str(row.get("loan_id") or "")
        if not loan_id.isdigit():
            continue
        if int(loan_id) % modulo == 0:
            sampled_rows.append(row)
            if len(sampled_rows) >= limit:
                break
    if not sampled_rows:
        sampled_rows = list(feature_rows)[:limit]
    return sampled_rows


def _band_label(value: float) -> str:
    if value >= 0.70:
        return "++"
    if value >= 0.30:
        return "+"
    if value <= -0.70:
        return "--"
    if value <= -0.30:
        return "-"
    return "."


def _compute_credit_history_age_years(clean_row: dict[str, Any]) -> float:
    issue_date = clean_row.get("issue_date")
    earliest_credit_line = clean_row.get("earliest_credit_line")
    if issue_date is None or earliest_credit_line is None:
        return 0.0
    months = (issue_date.year - earliest_credit_line.year) * 12 + (issue_date.month - earliest_credit_line.month)
    return max(months, 0) / 12.0


def _float_or_zero(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    numerator_value = _float_or_zero(numerator)
    denominator_value = _float_or_zero(denominator)
    if denominator_value == 0:
        return 0.0
    return numerator_value / denominator_value


def _safe_delta(current: Any, previous: Any) -> float:
    current_value = _float_or_zero(current)
    previous_value = _float_or_zero(previous)
    return current_value - previous_value


def _pearson(left_values: Sequence[float], right_values: Sequence[float]) -> float:
    if not left_values or not right_values or len(left_values) != len(right_values):
        return 0.0
    left_mean = mean(left_values)
    right_mean = mean(right_values)
    numerator = 0.0
    left_sum = 0.0
    right_sum = 0.0
    for left, right in zip(left_values, right_values, strict=True):
        left_delta = left - left_mean
        right_delta = right - right_mean
        numerator += left_delta * right_delta
        left_sum += left_delta * left_delta
        right_sum += right_delta * right_delta
    if left_sum == 0.0 or right_sum == 0.0:
        return 0.0
    return numerator / sqrt(left_sum * right_sum)
