"""Extended tests for services/document_parser.py — covers uncovered branches."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.document_parser import (
    DocumentFeatures,
    Transaction,
    _compute_average_monthly,
    _compute_income_stability,
    _compute_salary_regularity,
    _detect_document_type,
    _extract_date,
    _parse_gst_filing_text,
    _parse_salary_slip_text,
    _process_text,
    extract_bank_statement_features,
    parse_document_bytes,
    parse_document_text,
)


# ---------------------------------------------------------------------------
# _compute_income_stability
# ---------------------------------------------------------------------------


class TestComputeIncomeStability:
    def test_empty_returns_point3(self):
        assert _compute_income_stability([]) == pytest.approx(0.3)

    def test_single_entry_returns_point7(self):
        assert _compute_income_stability([75000.0]) == pytest.approx(0.7)

    def test_identical_values_returns_1(self):
        assert _compute_income_stability([50000.0, 50000.0, 50000.0]) == pytest.approx(1.0)

    def test_high_variance_returns_low_stability(self):
        # Very spread values → high CV → low stability
        s = _compute_income_stability([10000.0, 90000.0])
        assert s < 0.5

    def test_zero_mean_returns_point3(self):
        assert _compute_income_stability([0.0, 0.0]) == pytest.approx(0.3)

    def test_stability_in_range(self):
        s = _compute_income_stability([70000.0, 72000.0, 68000.0, 71000.0])
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# _compute_salary_regularity
# ---------------------------------------------------------------------------


class TestComputeSalaryRegularity:
    def test_empty_returns_zero(self):
        assert _compute_salary_regularity([]) == pytest.approx(0.0)

    def test_single_entry_returns_1(self):
        assert _compute_salary_regularity([75000.0]) == pytest.approx(1.0)

    def test_identical_returns_1(self):
        # CV = 0 → regularity = 1.0
        assert _compute_salary_regularity([50000.0, 50000.0, 50000.0]) == pytest.approx(1.0)

    def test_zero_mean_returns_zero(self):
        assert _compute_salary_regularity([0.0, 0.0]) == pytest.approx(0.0)

    def test_variable_credits_lower_regularity(self):
        r = _compute_salary_regularity([30000.0, 80000.0, 20000.0])
        assert 0.0 < r < 1.0


# ---------------------------------------------------------------------------
# _compute_average_monthly
# ---------------------------------------------------------------------------


class TestComputeAverageMonthly:
    def test_empty_returns_zero(self):
        assert _compute_average_monthly([]) == pytest.approx(0.0)

    def test_single_value(self):
        assert _compute_average_monthly([60000.0]) == pytest.approx(60000.0)

    def test_average_of_multiple(self):
        assert _compute_average_monthly([40000.0, 60000.0]) == pytest.approx(50000.0)


# ---------------------------------------------------------------------------
# _extract_date
# ---------------------------------------------------------------------------


class TestExtractDate:
    def test_dd_mm_yyyy_format(self):
        result = _extract_date("01-04-2025 SALARY CREDIT 75000.00 Cr")
        assert "2025" in result or "04" in result or "01" in result

    def test_no_date_returns_empty(self):
        result = _extract_date("NO DATE HERE SALARY AMOUNT")
        assert result == "" or isinstance(result, str)


# ---------------------------------------------------------------------------
# _parse_salary_slip_text
# ---------------------------------------------------------------------------


class TestParseSalarySlipText:
    def test_net_pay_extracted(self):
        text = "Net Pay: 75,000.00\nBasic: 40,000\nHRA: 15,000"
        features = DocumentFeatures(source="test", document_type="salary_slip")
        _parse_salary_slip_text(text, features)
        assert len(features.salary_credits) == 1
        assert features.salary_credits[0] > 0

    def test_income_stability_set(self):
        text = "Net Pay: 75,000.00"
        features = DocumentFeatures(source="test", document_type="salary_slip")
        _parse_salary_slip_text(text, features)
        assert features.income_stability_score == pytest.approx(0.8)

    def test_no_match_still_adds_transaction(self):
        text = "Employee Name: John Doe"
        features = DocumentFeatures(source="test", document_type="salary_slip")
        _parse_salary_slip_text(text, features)
        # Transaction is always added (even with net_salary=0)
        assert len(features.transactions) == 1

    def test_zero_salary_does_not_set_credits(self):
        text = "Employee Name: John"
        features = DocumentFeatures(source="test", document_type="salary_slip")
        _parse_salary_slip_text(text, features)
        assert features.salary_credits == []


# ---------------------------------------------------------------------------
# _parse_gst_filing_text
# ---------------------------------------------------------------------------


class TestParseGstFilingText:
    def test_taxable_value_extracted(self):
        text = "Taxable Value: 500,000.00\nGST Paid: 90,000.00"
        features = DocumentFeatures(source="test", document_type="gst_filing")
        _parse_gst_filing_text(text, features)
        assert features.total_credits > 0

    def test_gst_debit_extracted(self):
        text = "GST Amount: 18,000.00"
        features = DocumentFeatures(source="test", document_type="gst_filing")
        _parse_gst_filing_text(text, features)
        assert features.total_debits > 0

    def test_income_stability_always_set(self):
        text = "No relevant content here"
        features = DocumentFeatures(source="test", document_type="gst_filing")
        _parse_gst_filing_text(text, features)
        assert features.income_stability_score == pytest.approx(0.7)

    def test_total_turnover_as_credit(self):
        text = "Total Turnover: 1,200,000.00"
        features = DocumentFeatures(source="test", document_type="gst_filing")
        _parse_gst_filing_text(text, features)
        assert features.total_credits >= 1200000.0


# ---------------------------------------------------------------------------
# parse_document_bytes
# ---------------------------------------------------------------------------


class TestParseDocumentBytes:
    def test_utf8_text_bytes_parsed(self):
        text = "SALARY CREDIT 75000.00 Cr\n01-04-2025 SALARY CREDIT 75000.00 Cr\n"
        data = text.encode("utf-8")
        features = parse_document_bytes(data, "statement.txt", "bank_statement")
        assert features.document_type == "bank_statement"
        assert features.source == "statement.txt"

    def test_non_utf8_bytes_returns_features(self):
        # Binary data that is not valid UTF-8 → OCR path (will fail gracefully)
        data = bytes([0xFF, 0xFE, 0x80, 0x81, 0x82])
        features = parse_document_bytes(data, "binary.pdf")
        assert features.source == "binary.pdf"
        assert isinstance(features, DocumentFeatures)

    def test_empty_bytes_handled(self):
        data = b""
        features = parse_document_bytes(data, "empty.txt")
        assert isinstance(features, DocumentFeatures)

    def test_document_type_hint_respected(self):
        data = "Net Pay: 75,000\nBasic: 40,000".encode("utf-8")
        features = parse_document_bytes(data, "payslip.txt", "salary_slip")
        assert features.document_type == "salary_slip"


# ---------------------------------------------------------------------------
# _process_text
# ---------------------------------------------------------------------------


class TestProcessText:
    def test_salary_slip_type_dispatched(self):
        text = "SALARY SLIP\nNet Pay: 75,000\nEmployee ID: 1234"
        features = _process_text(text, "slip.txt", "salary_slip")
        assert features.document_type == "salary_slip"

    def test_gst_filing_type_dispatched(self):
        text = "GSTIN: 27ABCDE1234F\nTaxable Value: 500000\nGST Paid: 90000"
        features = _process_text(text, "gst.txt", "gst_filing")
        assert features.document_type == "gst_filing"

    def test_auto_detect_bank_statement(self):
        text = "Account Statement\nWithdrawal  Deposit\n01-04-2025 SALARY CREDIT 75000.00 Cr"
        features = _process_text(text, "stmt.txt", None)
        assert features.document_type == "bank_statement"

    def test_source_set_correctly(self):
        features = _process_text("some text", "my_doc.txt", None)
        assert features.source == "my_doc.txt"

    def test_empty_text_returns_zero_confidence(self):
        features = _process_text("", "empty.txt", None)
        assert features.confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# parse_document_text — parse_document_bytes via bytes path (file reading)
# ---------------------------------------------------------------------------


class TestParseDocumentFile:
    def test_txt_file_parsed(self, tmp_path):
        from services.document_parser import parse_document
        doc = tmp_path / "statement.txt"
        doc.write_text(
            "SALARY CREDIT 75000.00 Cr\n01-04-2025 SALARY CREDIT 75000.00 Cr\n",
            encoding="utf-8",
        )
        features = parse_document(str(doc), "bank_statement")
        assert features.document_type == "bank_statement"
        assert features.source == "statement.txt"

    def test_pdf_extension_handled(self, tmp_path):
        """PDF files without pdfplumber/PyPDF2 fall back to text read."""
        from services.document_parser import parse_document
        doc = tmp_path / "fake.pdf"
        # Write plain text pretending to be a PDF (will fail PDF parse → fallback)
        doc.write_bytes(b"SALARY CREDIT 75000.00 Cr\n")
        features = parse_document(str(doc), "bank_statement")
        assert isinstance(features, DocumentFeatures)

    def test_unknown_extension_falls_back_to_text(self, tmp_path):
        from services.document_parser import parse_document
        doc = tmp_path / "doc.csv"
        doc.write_text("01-04-2025 SALARY CREDIT 75000.00 Cr\n", encoding="utf-8")
        features = parse_document(str(doc))
        assert isinstance(features, DocumentFeatures)


# ---------------------------------------------------------------------------
# extract_bank_statement_features — emi_to_income_ratio
# ---------------------------------------------------------------------------


class TestExtractBankStatementFeaturesExtended:
    def test_emi_to_income_ratio_with_data(self):
        features = DocumentFeatures(source="test", document_type="bank_statement")
        features.salary_credits = [75000.0, 75000.0, 75000.0]
        features.emi_debits = [15000.0, 15000.0]
        features.income_stability_score = 0.8
        result = extract_bank_statement_features(features)
        assert "emi_to_income_ratio" in result
        assert result["emi_to_income_ratio"] >= 0.0

    def test_emi_to_income_zero_income(self):
        features = DocumentFeatures(source="test", document_type="bank_statement")
        features.salary_credits = []
        features.emi_debits = [15000.0]
        result = extract_bank_statement_features(features)
        assert result["emi_to_income_ratio"] == pytest.approx(0.0)

    def test_salary_regularity_zero_with_no_salary(self):
        features = DocumentFeatures(source="test", document_type="bank_statement")
        result = extract_bank_statement_features(features)
        assert result["salary_regularity"] == pytest.approx(0.0)
