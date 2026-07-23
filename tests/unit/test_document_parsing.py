from __future__ import annotations

from services.document_parser import (
    MOCK_GST_FILING,
    MOCK_HDFC_STATEMENT,
    MOCK_SALARY_SLIP,
    DocumentFeatures,
    Transaction,
    _detect_document_type,
    extract_bank_statement_features,
    parse_document_text,
)


def test_parse_hdfc_bank_statement() -> None:
    features = parse_document_text(MOCK_HDFC_STATEMENT, "bank_statement")
    assert features.document_type == "bank_statement"
    assert len(features.transactions) > 0
    assert features.total_credits > 0
    assert features.total_debits > 0
    assert features.average_balance > 0
    assert features.source == "<text>"


def test_parse_hdfc_salary_credits() -> None:
    features = parse_document_text(MOCK_HDFC_STATEMENT, "bank_statement")
    assert len(features.salary_credits) >= 3
    assert all(s > 0 for s in features.salary_credits)


def test_parse_hdfc_emi_detection() -> None:
    features = parse_document_text(MOCK_HDFC_STATEMENT, "bank_statement")
    assert len(features.emi_debits) >= 3
    assert all(e > 0 for e in features.emi_debits)


def test_parse_hdfc_bounce_count() -> None:
    features = parse_document_text(MOCK_HDFC_STATEMENT, "bank_statement")
    assert features.bounce_count >= 1


def test_parse_hdfc_closing_balance() -> None:
    features = parse_document_text(MOCK_HDFC_STATEMENT, "bank_statement")
    assert features.average_balance > 0


def test_parse_salary_slip() -> None:
    features = parse_document_text(MOCK_SALARY_SLIP, "salary_slip")
    assert features.document_type == "salary_slip"
    assert len(features.salary_credits) >= 1
    assert features.total_credits > 0
    assert features.income_stability_score > 0


def test_parse_salary_slip_net_pay() -> None:
    features = parse_document_text(MOCK_SALARY_SLIP, "salary_slip")
    assert 60000 <= features.salary_credits[0] <= 80000


def test_parse_gst_filing() -> None:
    features = parse_document_text(MOCK_GST_FILING, "gst_filing")
    assert features.document_type == "gst_filing"
    assert len(features.transactions) > 0
    assert features.total_credits > 0
    assert features.total_debits > 0


def test_parse_gst_turnover_detected() -> None:
    features = parse_document_text(MOCK_GST_FILING, "gst_filing")
    assert features.total_credits >= 500000


def test_extract_bank_statement_features_keys() -> None:
    features = parse_document_text(MOCK_HDFC_STATEMENT, "bank_statement")
    result = extract_bank_statement_features(features)
    expected_keys = {
        "income_stability_score",
        "average_monthly_inflow",
        "average_monthly_outflow",
        "bounce_count",
        "salary_regularity",
        "emi_to_income_ratio",
    }
    assert set(result.keys()) == expected_keys


def test_extract_income_stability_score() -> None:
    features = parse_document_text(MOCK_HDFC_STATEMENT, "bank_statement")
    result = extract_bank_statement_features(features)
    assert 0.0 <= result["income_stability_score"] <= 1.0


def test_extract_average_monthly_inflow() -> None:
    features = parse_document_text(MOCK_HDFC_STATEMENT, "bank_statement")
    result = extract_bank_statement_features(features)
    assert result["average_monthly_inflow"] > 0


def test_extract_bounce_count() -> None:
    features = parse_document_text(MOCK_HDFC_STATEMENT, "bank_statement")
    result = extract_bank_statement_features(features)
    assert result["bounce_count"] >= 1


def test_salary_regularity_computed() -> None:
    features = parse_document_text(MOCK_HDFC_STATEMENT, "bank_statement")
    result = extract_bank_statement_features(features)
    assert 0.0 <= result["salary_regularity"] <= 1.0


def test_detect_document_type_bank_statement() -> None:
    assert _detect_document_type("Account Statement\nTransactions\nWithdrawal\nDeposit") == "bank_statement"


def test_detect_document_type_salary_slip() -> None:
    assert _detect_document_type("Salary Slip\nEmployee Name\nEarnings\nDeductions\nNet Pay") == "salary_slip"


def test_detect_document_type_gst_filing() -> None:
    assert _detect_document_type("GSTIN: 27ABCDE1234F1Z5\nGST Return\nOutward Supply\nGST Paid") == "gst_filing"


def test_detect_document_type_gst_priority() -> None:
    text = "GSTIN 123\nGST Return\nOutward Supply\nalso statement and balance"
    assert _detect_document_type(text) == "gst_filing"


def test_empty_text_returns_low_confidence() -> None:
    features = parse_document_text("")
    assert features.confidence == 0.0
    assert "Empty document" in features.errors


def test_gibberish_text_returns_features() -> None:
    features = parse_document_text("x" * 1000)
    assert features.document_type is not None
    assert features.source == "<text>"
    assert len(features.transactions) == 0


def test_document_features_default_values() -> None:
    features = DocumentFeatures(source="test", document_type="bank_statement")
    assert features.transactions == []
    assert features.total_credits == 0.0
    assert features.total_debits == 0.0
    assert features.average_balance == 0.0
    assert features.salary_credits == []
    assert features.emi_debits == []
    assert features.bounce_count == 0
    assert features.income_stability_score == 0.5
    assert features.confidence == 1.0
    assert features.errors == []


def test_transaction_dataclass() -> None:
    txn = Transaction(date="01-04-2025", description="SALARY", amount=75000.0, type="credit")
    assert txn.date == "01-04-2025"
    assert txn.amount == 75000.0
    assert txn.type == "credit"


def test_salary_regularity_single_entry() -> None:
    features = parse_document_text("01-04-2025 SALARY CREDIT 75000.00 Cr\n", "bank_statement")
    result = extract_bank_statement_features(features)
    assert result["salary_regularity"] == 1.0


def test_parse_no_document_type_auto_detect() -> None:
    features = parse_document_text(MOCK_HDFC_STATEMENT)
    assert features.document_type == "bank_statement"


def test_custom_source_in_parse_text() -> None:
    from services.document_parser import _process_text

    features = _process_text(MOCK_HDFC_STATEMENT, "custom.txt", None)
    assert features.source == "custom.txt"


def test_extract_bank_statement_no_transactions() -> None:
    features = DocumentFeatures(source="empty", document_type="bank_statement")
    result = extract_bank_statement_features(features)
    assert result["income_stability_score"] == 0.5
    assert result["average_monthly_inflow"] == 0.0
    assert result["average_monthly_outflow"] == 0.0
    assert result["bounce_count"] == 0
    assert result["salary_regularity"] == 0.0


def test_income_stability_no_salary() -> None:
    features = parse_document_text("01-04-2025 RENT PAYMENT 20000.00 Dr\n", "bank_statement")
    assert features.income_stability_score == 0.3


def test_income_stability_single_salary() -> None:
    features = parse_document_text("01-04-2025 SALARY CREDIT 75000.00 Cr\n", "bank_statement")
    assert features.income_stability_score > 0.5
