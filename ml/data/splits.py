"""Time-based train, validation, and test split helpers for Lending Club data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable


@dataclass(frozen=True)
class TimeSplitBoundaries:
    """Working split boundaries for the 2007-2018Q4 Lending Club corpus."""

    train_end: date = date(2015, 12, 31)
    validation_end: date = date(2017, 12, 31)
    test_end: date = date(2018, 12, 31)


DEFAULT_TIME_SPLIT_BOUNDARIES = TimeSplitBoundaries()


def assign_time_split(
    issue_date: date,
    boundaries: TimeSplitBoundaries = DEFAULT_TIME_SPLIT_BOUNDARIES,
) -> str:
    """Return the split label for a single issue date."""

    if issue_date <= boundaries.train_end:
        return "train"
    if issue_date <= boundaries.validation_end:
        return "validation"
    if issue_date <= boundaries.test_end:
        return "test"
    return "holdout"


def split_rows_by_issue_date(
    rows: Iterable[dict[str, Any]],
    boundaries: TimeSplitBoundaries = DEFAULT_TIME_SPLIT_BOUNDARIES,
) -> dict[str, list[dict[str, Any]]]:
    """Partition cleaned or engineered rows into train/validation/test buckets."""

    buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
        "holdout": [],
    }
    for row in rows:
        issue_date = row.get("issue_date")
        if issue_date is None:
            continue
        split_name = assign_time_split(issue_date, boundaries)
        buckets[split_name].append(row)
    return buckets


def summarize_split_counts(
    split_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    """Return deterministic row counts per split."""

    return {split_name: len(rows) for split_name, rows in split_rows.items()}
