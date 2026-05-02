"""Dataset loading, cleaning, and profiling utilities for Lending Club data."""

from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Iterator, Sequence

DEFAULT_LENDING_CLUB_DATA_PATH = Path("ml/data/raw/accepted_2007_to_2018Q4.csv.gz")

MODELING_COLUMNS: tuple[str, ...] = (
    "id",
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "application_type",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "last_fico_range_low",
    "last_fico_range_high",
    "inq_last_6mths",
    "inq_last_12m",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_cur_bal",
    "total_bal_ex_mort",
    "total_rev_hi_lim",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "bc_util",
    "percent_bc_gt_75",
    "all_util",
    "il_util",
    "acc_open_past_24mths",
    "mort_acc",
    "pct_tl_nvr_dlq",
    "total_cu_tl",
    "open_rv_24m",
    "open_il_24m",
    "num_tl_90g_dpd_24m",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
)

DEFAULT_STATUS_MAP: dict[str, int] = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
    "Default": 1,
}

EXCLUDED_STATUSES: frozenset[str] = frozenset(
    {
        "",
        "Current",
        "In Grace Period",
        "Late (16-30 days)",
        "Late (31-120 days)",
    }
)


@dataclass(frozen=True)
class NumericSummary:
    """Simple descriptive summary for a numeric field."""

    count: int
    missing: int
    minimum: float
    median: float
    mean: float
    maximum: float


@dataclass(frozen=True)
class DatasetProfile:
    """High-level profiling summary for the cleaned modeling subset."""

    source_path: str
    total_rows: int
    individual_rows: int
    modeled_rows: int
    excluded_rows: int
    defaulted_rows: int
    non_defaulted_rows: int
    issue_date_min: str
    issue_date_max: str
    split_counts: dict[str, int]
    status_counts: dict[str, int]
    application_type_counts: dict[str, int]
    missing_issue_dates: int
    numeric_summaries: dict[str, NumericSummary]
    top_home_ownership: list[tuple[str, int]]
    top_verification_status: list[tuple[str, int]]
    top_purpose: list[tuple[str, int]]


def get_lending_club_data_path(env_var: str = "LENDING_CLUB_DATA_PATH") -> Path:
    """Return the configured dataset path, falling back to the canonical repo path."""

    raw_path = os.getenv(env_var)
    if raw_path:
        return Path(raw_path).expanduser()
    return DEFAULT_LENDING_CLUB_DATA_PATH


def resolve_lending_club_data_path(env_var: str = "LENDING_CLUB_DATA_PATH") -> Path:
    """Resolve the configured dataset path against the current working directory."""

    data_path = get_lending_club_data_path(env_var=env_var)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    return data_path.resolve()


def ensure_lending_club_data_path(env_var: str = "LENDING_CLUB_DATA_PATH") -> Path:
    """Return a resolved dataset path and raise a clear error when it is missing."""

    data_path = resolve_lending_club_data_path(env_var=env_var)
    if not data_path.exists():
        raise FileNotFoundError(
            "Lending Club dataset not found. "
            f"Set {env_var} or place the file at {DEFAULT_LENDING_CLUB_DATA_PATH}."
        )
    return data_path


def iter_raw_lending_club_rows(
    *,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    limit: int | None = None,
) -> Iterator[dict[str, str]]:
    """Yield raw CSV rows as dictionaries."""

    data_path = ensure_lending_club_data_path(env_var=env_var)
    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=1):
            if limit is not None and row_number > limit:
                break
            yield row


def clean_lending_club_row(
    raw_row: dict[str, str],
    *,
    status_map: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    """Normalize one Lending Club row into a deterministic modeling record."""

    status_lookup = status_map or DEFAULT_STATUS_MAP
    application_type = _clean_string(raw_row.get("application_type"))
    loan_status = _clean_string(raw_row.get("loan_status"))
    issue_date = _parse_month_year(raw_row.get("issue_d"))

    if application_type != "Individual":
        return None
    if issue_date is None:
        return None
    if loan_status not in status_lookup:
        return None

    annual_income = _parse_float(raw_row.get("annual_inc"))
    loan_amount = _parse_float(raw_row.get("loan_amnt"))
    term_months = _parse_term_months(raw_row.get("term"))
    if annual_income is None or annual_income <= 0 or loan_amount is None or term_months is None:
        return None

    monthly_income = annual_income / 12.0
    dti_pct = _sanitize_non_negative(_parse_float(raw_row.get("dti")))
    estimated_existing_emi = None if dti_pct is None else monthly_income * (dti_pct / 100.0)

    fico_low = _parse_float(raw_row.get("fico_range_low"))
    fico_high = _parse_float(raw_row.get("fico_range_high"))
    last_fico_low = _parse_float(raw_row.get("last_fico_range_low"))
    last_fico_high = _parse_float(raw_row.get("last_fico_range_high"))

    return {
        "loan_id": _clean_string(raw_row.get("id")),
        "issue_date": issue_date,
        "loan_status": loan_status,
        "defaulted": status_lookup[loan_status],
        "loan_amount": loan_amount,
        "funded_amount": _parse_float(raw_row.get("funded_amnt")),
        "term_months": term_months,
        "interest_rate_pct": _parse_float(raw_row.get("int_rate")),
        "installment": _parse_float(raw_row.get("installment")),
        "grade": _clean_string(raw_row.get("grade")),
        "sub_grade": _clean_string(raw_row.get("sub_grade")),
        "employment_length_years": _parse_employment_years(raw_row.get("emp_length")),
        "home_ownership": _clean_string(raw_row.get("home_ownership")),
        "annual_income": annual_income,
        "monthly_income": monthly_income,
        "verification_status": _clean_string(raw_row.get("verification_status")),
        "purpose": _clean_string(raw_row.get("purpose")),
        "dti_pct": dti_pct,
        "estimated_existing_emi": estimated_existing_emi,
        "delinq_2yrs": _parse_float(raw_row.get("delinq_2yrs")),
        "earliest_credit_line": _parse_month_year(raw_row.get("earliest_cr_line")),
        "fico_range_low": fico_low,
        "fico_range_high": fico_high,
        "fico_midpoint": _midpoint(fico_low, fico_high),
        "last_fico_range_low": last_fico_low,
        "last_fico_range_high": last_fico_high,
        "last_fico_midpoint": _midpoint(last_fico_low, last_fico_high),
        "inq_last_6mths": _parse_float(raw_row.get("inq_last_6mths")),
        "inq_last_12m": _parse_float(raw_row.get("inq_last_12m")),
        "open_acc": _parse_float(raw_row.get("open_acc")),
        "pub_rec": _parse_float(raw_row.get("pub_rec")),
        "revol_bal": _parse_float(raw_row.get("revol_bal")),
        "revol_util_pct": _parse_float(raw_row.get("revol_util")),
        "total_acc": _parse_float(raw_row.get("total_acc")),
        "collections_12_mths_ex_med": _parse_float(raw_row.get("collections_12_mths_ex_med")),
        "pub_rec_bankruptcies": _parse_float(raw_row.get("pub_rec_bankruptcies")),
        "tax_liens": _parse_float(raw_row.get("tax_liens")),
        "tot_cur_bal": _parse_float(raw_row.get("tot_cur_bal")),
        "total_bal_ex_mort": _parse_float(raw_row.get("total_bal_ex_mort")),
        "total_rev_hi_lim": _parse_float(raw_row.get("total_rev_hi_lim")),
        "total_bc_limit": _parse_float(raw_row.get("total_bc_limit")),
        "total_il_high_credit_limit": _parse_float(raw_row.get("total_il_high_credit_limit")),
        "bc_util_pct": _parse_float(raw_row.get("bc_util")),
        "percent_bc_gt_75": _parse_float(raw_row.get("percent_bc_gt_75")),
        "all_util_pct": _parse_float(raw_row.get("all_util")),
        "il_util_pct": _parse_float(raw_row.get("il_util")),
        "acc_open_past_24mths": _parse_float(raw_row.get("acc_open_past_24mths")),
        "mort_acc": _parse_float(raw_row.get("mort_acc")),
        "pct_tl_nvr_dlq": _parse_float(raw_row.get("pct_tl_nvr_dlq")),
        "total_cu_tl": _parse_float(raw_row.get("total_cu_tl")),
        "open_rv_24m": _parse_float(raw_row.get("open_rv_24m")),
        "open_il_24m": _parse_float(raw_row.get("open_il_24m")),
        "num_tl_90g_dpd_24m": _parse_float(raw_row.get("num_tl_90g_dpd_24m")),
        "mo_sin_old_rev_tl_op": _parse_float(raw_row.get("mo_sin_old_rev_tl_op")),
        "mo_sin_rcnt_rev_tl_op": _parse_float(raw_row.get("mo_sin_rcnt_rev_tl_op")),
    }


def iter_clean_lending_club_rows(
    *,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    limit: int | None = None,
    status_map: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield cleaned rows suitable for feature engineering and modeling."""

    for raw_row in iter_raw_lending_club_rows(env_var=env_var, limit=limit):
        cleaned_row = clean_lending_club_row(raw_row, status_map=status_map)
        if cleaned_row is not None:
            yield cleaned_row


def load_lending_club_data(
    *,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load cleaned modeling rows into memory."""

    return list(iter_clean_lending_club_rows(env_var=env_var, limit=limit))


def load_lending_club_dataframe(
    *,
    usecols: Iterable[str] | None = None,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    **read_csv_kwargs: Any,
):
    """Load the configured dataset with pandas for later training phases."""

    import pandas as pd

    data_path = ensure_lending_club_data_path(env_var=env_var)
    return pd.read_csv(data_path, usecols=usecols, **read_csv_kwargs)


def profile_lending_club_data(
    *,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    limit: int | None = None,
) -> DatasetProfile:
    """Compute a deterministic profile for the modeling subset."""

    from ml.data.splits import assign_time_split

    total_rows = 0
    individual_rows = 0
    missing_issue_dates = 0
    split_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    application_type_counts: Counter[str] = Counter()
    home_ownership_counts: Counter[str] = Counter()
    verification_status_counts: Counter[str] = Counter()
    purpose_counts: Counter[str] = Counter()
    numeric_values: dict[str, list[float]] = {
        "loan_amount": [],
        "annual_income": [],
        "monthly_income": [],
        "estimated_existing_emi": [],
        "dti_pct": [],
        "fico_midpoint": [],
        "revol_util_pct": [],
        "term_months": [],
    }

    modeled_row_count = 0
    defaulted_rows = 0
    issue_date_min: date | None = None
    issue_date_max: date | None = None

    for raw_row in iter_raw_lending_club_rows(env_var=env_var, limit=limit):
        total_rows += 1
        application_type = _clean_string(raw_row.get("application_type"))
        loan_status = _clean_string(raw_row.get("loan_status"))
        application_type_counts[application_type] += 1
        status_counts[loan_status] += 1

        if application_type == "Individual":
            individual_rows += 1
        if _parse_month_year(raw_row.get("issue_d")) is None:
            missing_issue_dates += 1

        cleaned_row = clean_lending_club_row(raw_row)
        if cleaned_row is None:
            continue

        modeled_row_count += 1
        defaulted_rows += int(cleaned_row["defaulted"])
        issue_date = cleaned_row["issue_date"]
        if issue_date_min is None or issue_date < issue_date_min:
            issue_date_min = issue_date
        if issue_date_max is None or issue_date > issue_date_max:
            issue_date_max = issue_date

        split_counts[assign_time_split(issue_date)] += 1
        home_ownership_counts[str(cleaned_row["home_ownership"])] += 1
        verification_status_counts[str(cleaned_row["verification_status"])] += 1
        purpose_counts[str(cleaned_row["purpose"])] += 1

        for key in numeric_values:
            value = cleaned_row.get(key)
            if isinstance(value, (int, float)):
                numeric_values[key].append(float(value))

    if modeled_row_count == 0:
        raise ValueError("No modeling rows available after applying the Phase 2 filters.")

    numeric_summaries = {
        key: _summarize_numeric_field(values, modeled_row_count) for key, values in numeric_values.items()
    }
    assert issue_date_min is not None
    assert issue_date_max is not None

    return DatasetProfile(
        source_path=str(ensure_lending_club_data_path(env_var=env_var)),
        total_rows=total_rows,
        individual_rows=individual_rows,
        modeled_rows=modeled_row_count,
        excluded_rows=total_rows - modeled_row_count,
        defaulted_rows=defaulted_rows,
        non_defaulted_rows=modeled_row_count - defaulted_rows,
        issue_date_min=issue_date_min.isoformat(),
        issue_date_max=issue_date_max.isoformat(),
        split_counts=dict(split_counts),
        status_counts=dict(status_counts),
        application_type_counts=dict(application_type_counts),
        missing_issue_dates=missing_issue_dates,
        numeric_summaries=numeric_summaries,
        top_home_ownership=home_ownership_counts.most_common(5),
        top_verification_status=verification_status_counts.most_common(5),
        top_purpose=purpose_counts.most_common(10),
    )


def write_data_quality_report(
    output_path: str | Path,
    *,
    env_var: str = "LENDING_CLUB_DATA_PATH",
    limit: int | None = None,
) -> Path:
    """Write a markdown quality report for the current modeling subset."""

    from ml.data.splits import DEFAULT_TIME_SPLIT_BOUNDARIES

    profile = profile_lending_club_data(env_var=env_var, limit=limit)
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    default_rate = profile.defaulted_rows / profile.modeled_rows if profile.modeled_rows else 0.0

    summary_lines = [
        "# Phase 2 Data Quality Report",
        "",
        f"Generated on 2026-05-02 from `{profile.source_path}`.",
        "",
        "## Scope",
        "",
        "- Raw corpus: Lending Club accepted loans.",
        "- Modeling filter: `application_type == Individual`.",
        "- Outcome filter: keep only terminal statuses that can be labeled deterministically as defaulted or non-defaulted.",
        "- Working split strategy: "
        f"train <= {DEFAULT_TIME_SPLIT_BOUNDARIES.train_end.isoformat()}, "
        f"validation <= {DEFAULT_TIME_SPLIT_BOUNDARIES.validation_end.isoformat()}, "
        f"test <= {DEFAULT_TIME_SPLIT_BOUNDARIES.test_end.isoformat()}.",
        "",
        "## Row Counts",
        "",
        f"- Total rows scanned: {profile.total_rows:,}",
        f"- Individual applications: {profile.individual_rows:,}",
        f"- Modeled rows after status/date filters: {profile.modeled_rows:,}",
        f"- Excluded rows: {profile.excluded_rows:,}",
        f"- Defaulted rows: {profile.defaulted_rows:,}",
        f"- Non-defaulted rows: {profile.non_defaulted_rows:,}",
        f"- Default rate: {default_rate:.2%}",
        f"- Issue date range: {profile.issue_date_min} to {profile.issue_date_max}",
        f"- Missing issue dates in raw data: {profile.missing_issue_dates:,}",
        f"- Split counts: {sorted(profile.split_counts.items())}",
        "",
        "## AuditLend-Mapped Numeric Ranges",
        "",
        "| Field | Count | Missing | Min | Median | Mean | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for field_name, stats in profile.numeric_summaries.items():
        summary_lines.append(
            "| "
            f"{field_name} | {stats.count:,} | {stats.missing:,} | "
            f"{stats.minimum:,.2f} | {stats.median:,.2f} | {stats.mean:,.2f} | {stats.maximum:,.2f} |"
        )

    summary_lines.extend(
        [
            "",
            "## Top Categorical Distributions",
            "",
            f"- Home ownership: {profile.top_home_ownership}",
            f"- Verification status: {profile.top_verification_status}",
            f"- Purpose: {profile.top_purpose}",
            "",
            "## Raw Status Snapshot",
            "",
            f"- Status counts seen before filtering: {sorted(profile.status_counts.items())}",
            f"- Application types seen before filtering: {sorted(profile.application_type_counts.items())}",
        ]
    )

    report_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return report_path


def _clean_string(value: str | None) -> str:
    return (value or "").strip()


def _parse_float(value: str | None) -> float | None:
    raw_value = _clean_string(value)
    if raw_value in {"", "nan", "NaN", "None"}:
        return None
    return float(raw_value)


def _sanitize_non_negative(value: float | None) -> float | None:
    if value is None:
        return None
    return max(value, 0.0)


def _parse_month_year(value: str | None) -> date | None:
    raw_value = _clean_string(value)
    if not raw_value:
        return None
    return datetime.strptime(raw_value, "%b-%Y").date()


def _parse_term_months(value: str | None) -> int | None:
    raw_value = _clean_string(value)
    if not raw_value:
        return None
    number = raw_value.split()[0]
    return int(number)


def _parse_employment_years(value: str | None) -> float | None:
    raw_value = _clean_string(value)
    if raw_value in {"", "n/a"}:
        return None
    if raw_value == "10+ years":
        return 10.0
    if raw_value == "< 1 year":
        return 0.5
    return float(raw_value.split()[0])


def _midpoint(low: float | None, high: float | None) -> float | None:
    if low is None and high is None:
        return None
    low_value = high if low is None else low
    high_value = low if high is None else high
    assert low_value is not None
    assert high_value is not None
    return (low_value + high_value) / 2.0


def _summarize_numeric_field(values: Sequence[float], total_rows: int) -> NumericSummary:
    if not values:
        return NumericSummary(
            count=0,
            missing=total_rows,
            minimum=0.0,
            median=0.0,
            mean=0.0,
            maximum=0.0,
        )

    return NumericSummary(
        count=len(values),
        missing=total_rows - len(values),
        minimum=min(values),
        median=median(values),
        mean=mean(values),
        maximum=max(values),
    )
