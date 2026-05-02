from __future__ import annotations

from datetime import date

from ml.data.splits import assign_time_split, split_rows_by_issue_date, summarize_split_counts


def test_assign_time_split_uses_prd_boundaries():
    assert assign_time_split(date(2016, 12, 31)) == "train"
    assert assign_time_split(date(2017, 1, 1)) == "validation"
    assert assign_time_split(date(2018, 6, 1)) == "test"
    assert assign_time_split(date(2019, 1, 1)) == "holdout"


def test_split_rows_by_issue_date_groups_rows_consistently():
    rows = [
        {"loan_id": "1", "issue_date": date(2016, 5, 1)},
        {"loan_id": "2", "issue_date": date(2017, 3, 1)},
        {"loan_id": "3", "issue_date": date(2018, 10, 1)},
        {"loan_id": "4", "issue_date": date(2019, 1, 1)},
    ]

    split_rows = split_rows_by_issue_date(rows)

    assert summarize_split_counts(split_rows) == {
        "train": 1,
        "validation": 1,
        "test": 1,
        "holdout": 1,
    }
