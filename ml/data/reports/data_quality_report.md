# Phase 2 Data Quality Report

Generated on 2026-05-02 from `/Users/purvansh/Desktop/Projects/AuditLend Intelligence Core (ALICe)/ml/data/raw/Lending Club Loan Data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`.

## Scope

- Raw corpus: Lending Club accepted loans.
- Modeling filter: `application_type == Individual`.
- Outcome filter: keep only terminal statuses that can be labeled deterministically as defaulted or non-defaulted.
- Working split strategy: train <= 2016-12-31, validation <= 2017-12-31, test <= 2018-12-31.

## Row Counts

- Total rows scanned: 2,260,701
- Individual applications: 2,139,958
- Modeled rows after status/date filters: 1,322,289
- Excluded rows: 938,412
- Defaulted rows: 263,010
- Non-defaulted rows: 1,059,279
- Default rate: 19.89%
- Issue date range: 2007-06-01 to 2018-12-01
- Missing issue dates in raw data: 33
- Split counts: [('test', 49230), ('train', 828950), ('validation', 444109)]

## AuditLend-Mapped Numeric Ranges

| Field | Count | Missing | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loan_amount | 1,322,289 | 0 | 500.00 | 12,000.00 | 14,319.64 | 40,000.00 |
| annual_income | 1,322,289 | 0 | 1,896.00 | 65,000.00 | 76,599.15 | 10,999,200.00 |
| monthly_income | 1,322,289 | 0 | 158.00 | 5,416.67 | 6,383.26 | 916,600.00 |
| estimated_existing_emi | 1,322,289 | 0 | 0.00 | 924.00 | 1,067.97 | 42,802.08 |
| dti_pct | 1,322,289 | 0 | 0.00 | 17.52 | 18.01 | 49.96 |
| fico_midpoint | 1,322,289 | 0 | 612.00 | 692.00 | 698.00 | 847.50 |
| revol_util_pct | 1,321,441 | 848 | 0.00 | 52.20 | 51.87 | 892.30 |
| term_months | 1,322,289 | 0 | 36.00 | 36.00 | 41.72 | 60.00 |

## Top Categorical Distributions

- Home ownership: [('MORTGAGE', 649378), ('RENT', 529785), ('OWN', 142611), ('ANY', 286), ('OTHER', 182)]
- Verification status: [('Source Verified', 514178), ('Verified', 408974), ('Not Verified', 399137)]
- Purpose: [('debt_consolidation', 765469), ('credit_card', 291186), ('home_improvement', 85672), ('other', 76851), ('major_purchase', 29055), ('small_business', 15341), ('medical', 15157), ('car', 14446), ('moving', 9335), ('vacation', 8953)]

## Raw Status Snapshot

- Status counts seen before filtering: [('', 33), ('Charged Off', 268559), ('Current', 878317), ('Default', 40), ('Does not meet the credit policy. Status:Charged Off', 761), ('Does not meet the credit policy. Status:Fully Paid', 1988), ('Fully Paid', 1076751), ('In Grace Period', 8436), ('Late (16-30 days)', 4349), ('Late (31-120 days)', 21467)]
- Application types seen before filtering: [('', 33), ('Individual', 2139958), ('Joint App', 120710)]
