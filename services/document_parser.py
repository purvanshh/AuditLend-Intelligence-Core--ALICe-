from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO


@dataclass
class Transaction:
    date: str
    description: str
    amount: float
    type: str  # "credit" | "debit"


@dataclass
class DocumentFeatures:
    source: str
    document_type: str  # "bank_statement" | "salary_slip" | "gst_filing"
    transactions: list[Transaction] = field(default_factory=list)
    total_credits: float = 0.0
    total_debits: float = 0.0
    average_balance: float = 0.0
    salary_credits: list[float] = field(default_factory=list)
    emi_debits: list[float] = field(default_factory=list)
    bounce_count: int = 0
    income_stability_score: float = 0.5
    confidence: float = 1.0
    errors: list[str] = field(default_factory=list)


HDFC_TXN_RE = re.compile(
    r"(\d{2}[/-]\d{2}[/-]\d{4}|\d{2}[/-]\d{2}[/-]\d{2})\s+"
    r"([A-Za-z0-9\s#/-]+?)\s+"
    r"([\d,]+\.\d{2})\s+"
    r"(Cr|Dr)",
    re.IGNORECASE,
)

ICICI_TXN_RE = re.compile(
    r"(\d{2}[/-]\d{2}[/-]\d{4}|\d{2}[/-]\d{2}[/-]\d{2})\s+"
    r"([A-Za-z0-9\s#/-]+?)\s+"
    r"([\d,]+\.\d{2})\s*$",
    re.MULTILINE,
)

SBI_TXN_RE = re.compile(
    r"(\d{2}[/-]\d{2}[/-]\d{4}|\d{2}[/-]\d{2}[/-]\d{2})\s+"
    r"([A-Za-z0-9\s#/-]+?)\s+"
    r"([\d,]+\.\d{2})\s+"
    r"(Cr|Dr)",
    re.IGNORECASE,
)

SALARY_KEYWORDS = [
    "salary", "sal", "payroll", "wages", "remuneration", "stipend",
    "salary credit", "sal credit",
]

EMI_KEYWORDS = [
    "emi", "loan repayment", "loan emi", "home loan", "car loan",
    "personal loan", "auto debit emi", "loan",
]

BOUNCE_KEYWORDS = [
    "bounce", "dishonour", "insufficient", "return", "unpaid",
    "cheque return", "payment returned", "nbounce",
]

BALANCE_KEYWORDS = [
    "closing balance", "balance", "available balance", "total balance",
]


def parse_document(file_path: str | Path, document_type: str | None = None) -> DocumentFeatures:
    """Parse a document and extract financial features.

    Supports PDF and image formats. Auto-detects document type from content
    if not specified.

    For PDF files, attempts to extract text using pdfplumber (preferred) or
    PyPDF2. For image files, attempts OCR via pytesseract. Falls back to
    plain text decoding for .txt files.

    Args:
        file_path: Path to the document file.
        document_type: Optional hint ("bank_statement", "salary_slip", "gst_filing").
                       Auto-detected from text content if not provided.

    Returns:
        DocumentFeatures with parsed transactions and financial metrics.
    """
    path = Path(file_path)
    text = _extract_text_from_file(path)
    return _process_text(text, path.name, document_type)


def parse_document_bytes(
    data: bytes,
    filename: str = "upload",
    document_type: str | None = None,
) -> DocumentFeatures:
    """Parse a document from raw bytes.

    Tries to decode as UTF-8 text first. If that fails, attempts OCR via
    pytesseract (requires PIL and pytesseract installed).

    Args:
        data: Raw file bytes.
        filename: Original filename for source tracking.
        document_type: Optional document type hint.

    Returns:
        DocumentFeatures with parsed transactions and financial metrics.
    """
    text: str | None = None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = _ocr_bytes(data, filename)

    if text is None:
        return DocumentFeatures(
            source=filename,
            document_type=document_type or "unknown",
            confidence=0.0,
            errors=["Could not extract text from bytes"],
        )

    return _process_text(text, filename, document_type)


def parse_document_text(text: str, document_type: str | None = None) -> DocumentFeatures:
    """Parse pre-extracted text from a document.

    Useful for testing or when text has already been extracted via OCR/PDF
    parsing externally.

    Args:
        text: The extracted text content.
        document_type: Optional document type hint.

    Returns:
        DocumentFeatures with parsed transactions and financial metrics.
    """
    return _process_text(text, "<text>", document_type)


def extract_bank_statement_features(features: DocumentFeatures) -> dict[str, Any]:
    """Convert parsed bank statement into model-ready features.

    Returns a dictionary with:
        - income_stability_score
        - average_monthly_inflow
        - average_monthly_outflow
        - bounce_count
        - salary_regularity (coefficient of variation of salary credits)
        - emi_to_income_ratio
    """
    salary_regularity = _compute_salary_regularity(features.salary_credits)
    avg_monthly_inflow = _compute_average_monthly(features.salary_credits)
    avg_monthly_outflow = _compute_average_monthly(features.emi_debits)

    emi_to_income_ratio = 0.0
    if avg_monthly_inflow > 0 and features.emi_debits:
        emi_to_income_ratio = round(sum(features.emi_debits) / len(features.emi_debits) / avg_monthly_inflow, 4)

    return {
        "income_stability_score": round(features.income_stability_score, 4),
        "average_monthly_inflow": round(avg_monthly_inflow, 2),
        "average_monthly_outflow": round(avg_monthly_outflow, 2),
        "bounce_count": features.bounce_count,
        "salary_regularity": round(salary_regularity, 4),
        "emi_to_income_ratio": round(emi_to_income_ratio, 4),
    }


def _process_text(text: str, source: str, document_type: str | None) -> DocumentFeatures:
    if not text.strip():
        return DocumentFeatures(
            source=source,
            document_type=document_type or "unknown",
            confidence=0.0,
            errors=["Empty document"],
        )

    if document_type is None:
        document_type = _detect_document_type(text)

    features = DocumentFeatures(source=source, document_type=document_type)

    if document_type == "bank_statement":
        _parse_bank_statement_text(text, features)
    elif document_type == "salary_slip":
        _parse_salary_slip_text(text, features)
    elif document_type == "gst_filing":
        _parse_gst_filing_text(text, features)

    _compute_confidence(features)
    return features


def _detect_document_type(text: str) -> str:
    """Heuristic document type detection from text content."""
    text_lower = text.lower()

    gst_indicators = [
        "gstin", "gst", "gst return", "gstr-", "tax period", "gst filing",
        "outward supply", "inward supply", "gst paid",
    ]
    gst_score = sum(1 for kw in gst_indicators if kw in text_lower)

    salary_indicators = [
        "salary slip", "payslip", "pay slip", "salary statement",
        "earnings", "deductions", "net pay", "employee name",
        "pan number", "uan number", "days worked",
    ]
    salary_score = sum(1 for kw in salary_indicators if kw in text_lower)

    bank_indicators = [
        "statement", "account statement", "transaction", "withdrawal",
        "deposit", "closing balance", "opening balance", "narrative",
        "cheque", "chq", "neft", "imps", "rtgs", "upi",
    ]
    bank_score = sum(1 for kw in bank_indicators if kw in text_lower)

    if gst_score >= salary_score and gst_score >= bank_score and gst_score >= 2:
        return "gst_filing"
    if salary_score >= bank_score and salary_score >= 1:
        return "salary_slip"
    if bank_score >= 1:
        return "bank_statement"

    return "bank_statement"


def _parse_bank_statement_text(text: str, features: DocumentFeatures) -> None:
    lines = text.strip().split("\n")
    balances: list[float] = []
    salary_amounts: list[float] = []
    emi_amounts: list[float] = []
    bounce_count = 0

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        amount_match = re.search(r"([\d,]+\.\d{2})", line_stripped)
        if not amount_match:
            continue

        amount_str = amount_match.group(1).replace(",", "")
        try:
            amount = float(amount_str)
        except ValueError:
            continue

        lower_line = line_stripped.lower()

        is_credit = bool(re.search(r"\b(cr|credit)\b", lower_line, re.IGNORECASE))
        is_debit = bool(re.search(r"\b(dr|debit|withdrawal)\b", lower_line, re.IGNORECASE))

        txn_type = "credit" if is_credit else ("debit" if is_debit else None)

        if txn_type is None:
            if re.match(r"^\d{2}[/-]\d{2}[/-]\d{2,4}", line_stripped):
                txn_type = "debit"
            else:
                continue

        features.transactions.append(
            Transaction(
                date=_extract_date(line_stripped),
                description=line_stripped[:80],
                amount=amount,
                type=txn_type,
            )
        )

        if txn_type == "credit":
            features.total_credits += amount
            if any(kw in lower_line for kw in SALARY_KEYWORDS):
                salary_amounts.append(amount)
        else:
            features.total_debits += amount
            if any(kw in lower_line for kw in EMI_KEYWORDS):
                emi_amounts.append(amount)

        if any(kw in lower_line for kw in BOUNCE_KEYWORDS):
            bounce_count += 1

        if any(kw in lower_line for kw in BALANCE_KEYWORDS):
            balances.append(amount)

    features.salary_credits = salary_amounts
    features.emi_debits = emi_amounts
    features.bounce_count = bounce_count

    if balances:
        features.average_balance = round(sum(balances) / len(balances), 2)
    elif features.transactions:
        net = features.total_credits - features.total_debits
        features.average_balance = round(net / max(len(features.transactions), 1), 2)

    features.income_stability_score = _compute_income_stability(salary_amounts)


def _parse_salary_slip_text(text: str, features: DocumentFeatures) -> None:
    text_lower = text.lower()
    earnings_patterns = [
        r"(?:basic|da|hra|conveyance|medical|special|allowance|dearness|pf|gratuity|bonus|incentive)\s*:?\s*([\d,]+\.?\d*)",
        r"(?:earnings|income|gross pay|total earnings)\s*:?\s*([\d,]+\.?\d*)",
        r"(?:net pay|net salary|take home|amount paid)\s*:?\s*([\d,]+\.?\d*)",
    ]

    amounts_found: list[float] = []
    for pattern in earnings_patterns:
        for match in re.finditer(pattern, text_lower):
            try:
                amounts_found.append(float(match.group(1).replace(",", "")))
            except ValueError:
                continue

    net_salary = 0.0
    if amounts_found:
        net_salary = max(amounts_found)

    if net_salary > 0:
        features.salary_credits = [net_salary]
        features.total_credits = net_salary
        features.income_stability_score = 0.8

    features.transactions.append(
        Transaction(
            date=_extract_date(text) or "",
            description=f"Salary slip: net pay {net_salary}",
            amount=net_salary,
            type="credit",
        )
    )


def _parse_gst_filing_text(text: str, features: DocumentFeatures) -> None:
    text_lower = text.lower()
    txn_patterns = [
        (r"(?:taxable value|assessable value|invoice value)\s*:?\s*([\d,]+\.?\d*)", "credit"),
        (r"(?:gst|tax|cess)\s*(?:paid|amount|liability)\s*:?\s*([\d,]+\.?\d*)", "debit"),
        (r"(?:total turnover|gross turnover|total sales)\s*:?\s*([\d,]+\.?\d*)", "credit"),
        (r"(?:output tax|outward)\s*:?\s*([\d,]+\.?\d*)", "credit"),
        (r"(?:input tax credit|itc)\s*:?\s*([\d,]+\.?\d*)", "debit"),
    ]

    for pattern, txn_type in txn_patterns:
        for match in re.finditer(pattern, text_lower):
            try:
                amount = float(match.group(1).replace(",", ""))
            except ValueError:
                continue

            features.transactions.append(
                Transaction(
                    date=_extract_date(text) or "",
                    description=match.group(0)[:80],
                    amount=amount,
                    type=txn_type,
                )
            )
            if txn_type == "credit":
                features.total_credits += amount
            else:
                features.total_debits += amount

    features.income_stability_score = 0.7


def _compute_income_stability(salary_amounts: list[float]) -> float:
    if not salary_amounts:
        return 0.3

    if len(salary_amounts) == 1:
        return 0.7

    mean_salary = sum(salary_amounts) / len(salary_amounts)
    if mean_salary == 0:
        return 0.3

    variance = sum((s - mean_salary) ** 2 for s in salary_amounts) / len(salary_amounts)
    cv = math.sqrt(variance) / mean_salary

    stability = max(0.0, min(1.0, 1.0 - cv))
    return round(stability, 4)


def _compute_salary_regularity(salary_credits: list[float]) -> float:
    if not salary_credits:
        return 0.0
    if len(salary_credits) == 1:
        return 1.0

    mean_s = sum(salary_credits) / len(salary_credits)
    if mean_s == 0:
        return 0.0

    variance = sum((s - mean_s) ** 2 for s in salary_credits) / len(salary_credits)
    cv = math.sqrt(variance) / mean_s
    if cv == 0:
        return 1.0
    return round(1.0 / (1.0 + cv), 4)


def _compute_average_monthly(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _compute_confidence(features: DocumentFeatures) -> None:
    text_quality = 1.0
    if features.errors:
        text_quality -= 0.2 * len(features.errors)

    transaction_richness = 0.0
    if features.transactions:
        transaction_richness = min(1.0, len(features.transactions) / 20)

    balance_factor = 0.5 if features.average_balance > 0 else 0.0

    features.confidence = round(
        max(0.0, min(1.0, (text_quality * 0.4 + transaction_richness * 0.3 + balance_factor * 0.3))),
        4,
    )


def _extract_date(text: str) -> str:
    date_match = re.search(r"(\d{2}[/-]\d{2}[/-]\d{4}|\d{2}[/-]\d{2}[/-]\d{2})", text)
    if date_match:
        return date_match.group(1)
    return ""


def _extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text("utf-8", errors="replace")

    text: str | None = None
    if suffix in (".pdf",):
        text = _extract_pdf_text(path)

    if text is None and suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        text = _ocr_image(path)

    if text is None:
        try:
            text = path.read_text("utf-8", errors="replace")
        except Exception:
            pass

    return text or ""


def _extract_pdf_text(path: Path) -> str | None:
    text = None
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        if text and text.strip():
            return text
    except ImportError:
        pass
    except Exception:
        pass

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        text = "".join(page.extract_text() or "" for page in reader.pages)
        if text and text.strip():
            return text
    except ImportError:
        pass
    except Exception:
        pass

    return None


def _ocr_image(path: Path) -> str | None:
    try:
        from PIL import Image

        import pytesseract

        image = Image.open(str(path))
        text = pytesseract.image_to_string(image)
        return text if text.strip() else None
    except ImportError:
        pass
    except Exception:
        pass
    return None


def _ocr_bytes(data: bytes, filename: str) -> str | None:
    try:
        from PIL import Image

        import pytesseract

        import io

        image = Image.open(io.BytesIO(data))
        text = pytesseract.image_to_string(image)
        return text if text.strip() else None
    except ImportError:
        pass
    except Exception:
        pass
    return None


MOCK_HDFC_STATEMENT = """HDFC BANK
SAVINGS ACCOUNT STATEMENT

Account No: xxxxxxxx1234
Period: 01-Apr-2025 to 30-Jun-2025

Date,Narration,Chq/Ref No,Value Date,Withdrawal Amt,Deposit Amt,Closing Balance
01-04-2025,OPENING BALANCE,,,,,50000.00
05-04-2025,SALARY CREDIT ACH/XYZ/12345,,05-04-2025,,75000.00,125000.00
06-04-2025,ATM WITHDRAWAL,123456,06-04-2025,5000.00,,120000.00
10-04-2025,UPI PAYMENT - ZOMATO,,10-04-2025,1200.00,,118800.00
12-04-2025,EMI - HOME LOAN ACH/HDFC/99887,,12-04-2025,25000.00,,93800.00
15-04-2025,NEFT - RENT PAYMENT,,15-04-2025,20000.00,,73800.00
20-04-2025,CREDIT CARD PAYMENT,,20-04-2025,15000.00,,58800.00
01-05-2025,SALARY CREDIT ACH/XYZ/12346,,01-05-2025,,75000.00,133800.00
03-05-2025,UPI PAYMENT - AMAZON,,03-05-2025,3500.00,,130300.00
10-05-2025,EMI - HOME LOAN ACH/HDFC/99887,,10-05-2025,25000.00,,105300.00
12-05-2025,BILL PAYMENT - ELECTRICITY,,12-05-2025,3200.00,,102100.00
20-05-2025,ATM WITHDRAWAL,789012,20-05-2025,10000.00,,92100.00
25-05-2025,CHQ RETURN - BOUNCE - INSUFFICIENT FUNDS,345678,25-05-2025,500.00,,91600.00
01-06-2025,SALARY CREDIT ACH/XYZ/12347,,01-06-2025,,80000.00,171600.00
05-06-2025,UPI PAYMENT - SWIGGY,,05-06-2025,450.00,,171150.00
12-06-2025,EMI - HOME LOAN ACH/HDFC/99887,,12-06-2025,25000.00,,146150.00
18-06-2025,NEFT - RENT PAYMENT,,18-06-2025,20000.00,,126150.00
25-06-2025,INTERNET BANKING TRANSFER,,25-06-2025,5000.00,,121150.00
30-06-2025,CLOSING BALANCE,,,,,121150.00
"""

MOCK_SALARY_SLIP = """SALARY SLIP - APRIL 2025
ABC Corp Pvt Ltd

Employee Name: Rajesh Kumar
PAN: ABCDE1234F
UAN: 123456789012
Days Worked: 30

Earnings:
  Basic: 35000.00
  HRA: 17500.00
  Conveyance: 8000.00
  Medical: 1250.00
  Special Allowance: 13250.00
  Gross Pay: 75000.00

Deductions:
  PF: 3600.00
  Professional Tax: 200.00
  Income Tax: 5000.00
  Total Deductions: 8800.00

Net Pay: 66200.00
"""

MOCK_GST_FILING = """GSTR-3B RETURN FOR APRIL 2025
GSTIN: 27ABCDE1234F1Z5
Period: Apr 2025

Outward Supply:
  Taxable Value: 500000.00
  CGST: 45000.00
  SGST: 45000.00

Inward Supply:
  Taxable Value: 200000.00
  ITC Available: 36000.00

Net GST Paid: 54000.00
Total Turnover: 500000.00
"""
