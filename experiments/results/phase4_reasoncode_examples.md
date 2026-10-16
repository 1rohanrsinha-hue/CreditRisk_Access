# Phase 4 worked reason-code examples (rep 0; anonymised: amounts rounded, no identifiers)

_Codes = largest risk-increasing contributions (positive = raises log-odds of default). Not a statement of regulatory compliance._

## Example 1: overlap 3/3

Feature values: loan_amount=8500, annual_income=20000, debt_to_income=19.2, delinquencies=0.0, num_credit_lines=7.0, total_credit_lines=10.0, loan_purpose=credit_card, home_ownership=RENT, public_records=0.0, credit_inquiries=1.0, mortgage_accounts=0.0, collections=0.0, employment_length=nan, emp_length_missing=1.0, interest_rate=15.6, fico_mean=662.0

| rank | EBM code | contribution | tuned LightGBM code | contribution |
|---|---|---|---|---|
| 1 | Interest rate assigned to the loan is high (interest_rate) | +0.435 | Interest rate assigned to the loan is high (interest_rate) | +0.397 |
| 2 | Annual income is low relative to comparable borrowers (annual_income) | +0.314 | Annual income is low relative to comparable borrowers (annual_income) | +0.305 |
| 3 | Employment length not reported (emp_length_missing) | +0.311 | Employment length not reported (emp_length_missing) | +0.276 |

## Example 2: overlap 2/3

Feature values: loan_amount=19000, annual_income=65000, debt_to_income=5.6, delinquencies=0.0, num_credit_lines=10.0, total_credit_lines=41.0, loan_purpose=house, home_ownership=MORTGAGE, public_records=0.0, credit_inquiries=1.0, mortgage_accounts=5.0, collections=0.0, employment_length=5.0, emp_length_missing=0.0, interest_rate=15.9, fico_mean=667.0

| rank | EBM code | contribution | tuned LightGBM code | contribution |
|---|---|---|---|---|
| 1 | Interest rate assigned to the loan is high (interest_rate) | +0.472 | Interest rate assigned to the loan is high (interest_rate) | +0.490 |
| 2 | Requested loan amount is high (loan_amount) | +0.208 | Requested loan amount is high (loan_amount) | +0.208 |
| 3 | Stated purpose of the loan (loan_purpose) | +0.119 | Credit score is lower than comparable borrowers (fico_mean) | +0.113 |

## Example 3: overlap 2/3

Feature values: loan_amount=29000, annual_income=65000, debt_to_income=15.5, delinquencies=0.0, num_credit_lines=13.0, total_credit_lines=27.0, loan_purpose=debt_consolidation, home_ownership=RENT, public_records=0.0, credit_inquiries=0.0, mortgage_accounts=0.0, collections=0.0, employment_length=10.0, emp_length_missing=0.0, interest_rate=18.8, fico_mean=682.0

| rank | EBM code | contribution | tuned LightGBM code | contribution |
|---|---|---|---|---|
| 1 | Interest rate assigned to the loan is high (interest_rate) | +0.694 | Interest rate assigned to the loan is high (interest_rate) | +0.694 |
| 2 | Requested loan amount is high (loan_amount) | +0.244 | Requested loan amount is high (loan_amount) | +0.193 |
| 3 | Housing status (rent/own/mortgage) (home_ownership) | +0.103 | Credit score is lower than comparable borrowers (fico_mean) | +0.088 |

## Example 4: overlap 1/3

Feature values: loan_amount=7500, annual_income=50000, debt_to_income=19.3, delinquencies=5.0, num_credit_lines=14.0, total_credit_lines=28.0, loan_purpose=credit_card, home_ownership=RENT, public_records=0.0, credit_inquiries=1.0, mortgage_accounts=0.0, collections=0.0, employment_length=2.0, emp_length_missing=0.0, interest_rate=9.8, fico_mean=667.0

| rank | EBM code | contribution | tuned LightGBM code | contribution |
|---|---|---|---|---|
| 1 | Recent delinquencies on credit accounts (delinquencies) | +0.124 | Credit score is lower than comparable borrowers (fico_mean) | +0.152 |
| 2 | Credit score is lower than comparable borrowers (fico_mean) | +0.122 | Number of open credit accounts is unusual (num_credit_lines) | +0.082 |
| 3 | Housing status (rent/own/mortgage) (home_ownership) | +0.103 | Debt payments are high relative to income (debt_to_income) | +0.074 |

## Example 5: overlap 0/3

Feature values: loan_amount=12000, annual_income=130000, debt_to_income=13.9, delinquencies=1.0, num_credit_lines=10.0, total_credit_lines=25.0, loan_purpose=home_improvement, home_ownership=MORTGAGE, public_records=1.0, credit_inquiries=0.0, mortgage_accounts=3.0, collections=0.0, employment_length=6.0, emp_length_missing=0.0, interest_rate=8.2, fico_mean=697.0

| rank | EBM code | contribution | tuned LightGBM code | contribution |
|---|---|---|---|---|
| 1 | Stated purpose of the loan (loan_purpose) | +0.017 | Derogatory public records on file (public_records) | +0.030 |
| 2 | Requested loan amount is high (loan_amount) | +0.013 | Credit score is lower than comparable borrowers (fico_mean) | +0.025 |
| 3 | Total number of credit accounts is unusual (total_credit_lines) | +0.010 | Length of employment is short (employment_length) | +0.008 |
