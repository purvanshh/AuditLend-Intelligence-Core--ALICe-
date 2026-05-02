from __future__ import annotations

from pathlib import Path

import pytest

from ml.data.ingestion import (
    DEFAULT_LENDING_CLUB_DATA_PATH,
    ensure_lending_club_data_path,
    resolve_lending_club_data_path,
)


def test_resolve_lending_club_data_path_uses_default_when_env_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("LENDING_CLUB_DATA_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    expected = (tmp_path / DEFAULT_LENDING_CLUB_DATA_PATH).resolve()

    assert resolve_lending_club_data_path() == expected


def test_ensure_lending_club_data_path_respects_env_override(monkeypatch, tmp_path):
    dataset_path = tmp_path / "custom-dataset.csv"
    dataset_path.write_text("loan_amnt\n1000\n", encoding="utf-8")
    monkeypatch.setenv("LENDING_CLUB_DATA_PATH", str(dataset_path))

    assert ensure_lending_club_data_path() == dataset_path.resolve()


def test_ensure_lending_club_data_path_raises_clear_error_when_missing(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing.csv.gz"
    monkeypatch.setenv("LENDING_CLUB_DATA_PATH", str(missing_path))

    with pytest.raises(FileNotFoundError) as exc_info:
        ensure_lending_club_data_path()

    assert "LENDING_CLUB_DATA_PATH" in str(exc_info.value)
    assert str(DEFAULT_LENDING_CLUB_DATA_PATH) in str(exc_info.value)
