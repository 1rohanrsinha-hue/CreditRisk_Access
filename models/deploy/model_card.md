# Model card: calibrated LightGBM credit-default PD (research prototype)

**Not for credit decisions.** This is a research prototype that accompanies a journal study. It has not been validated for any
lending decision, and whether any output satisfies adverse-action-notice requirements is a legal question outside the study.

## Intended use
Illustrating calibrated probability of default (PD) and review tiers on public Lending Club data. Tiers are *review* tiers, not
accept/reject cutoffs: on mature vintages no threshold rule earned more than approving everyone (study finding H5).

## Data
- Lending Club accepted loans 2007-2018Q4; strict label (Fully Paid = 0, Charged Off/Default = 1).
- Mature population D3: issue years 2010-2013 (any term) and 2014-2015 (36-month only), 667,024 loans, prevalence 0.1483 (training split).
- Training split 533,619 loans (issued 2010-01-01 to 2015-12-01); held-out test split 133,405 (stratified random 20%, seed 42).
- Funded loans only: rejected applicants are not observed.

## Model
- Tuned LightGBM (unweighted), hyper-parameters = median over the 5 outer folds of the nested Optuna search (integers rounded): `{"n_estimators": 408, "learning_rate": 0.02206251304554856, "num_leaves": 12, "min_child_samples": 109, "subsample": 0.7882305371173077, "colsample_bytree": 0.5922110875642564, "reg_lambda": 0.07243067561661562, "subsample_freq": 1, "max_depth": -1}`.
- Calibration: isotonic regression fitted on 5-fold out-of-fold scores of the training split (fresh pipeline per fold). Platt is available with `--calibrator platt`.
- Companion: Explainable Boosting Machine (10 interactions); reason codes are its exact additive term contributions (interactions split equally between their two features).
- Features (16, application-time): loan_amount, annual_income, debt_to_income, delinquencies, num_credit_lines, total_credit_lines, loan_purpose, home_ownership, public_records, credit_inquiries, mortgage_accounts, collections, employment_length, emp_length_missing, interest_rate, fico_mean. `interest_rate` is the lender's own risk price (see limitations). `fico_mean` is the mean of the reported FICO range.

## Held-out performance (test split, 95% bootstrap CIs, 1,000 resamples, in-time)
| metric | value |
|---|---|
| ROC-AUC (calibrated PD) | 0.6898 [0.6859, 0.6937] |
| PR-AUC | 0.2680 [0.2626, 0.2734] |
| Brier (calibrated / raw) | 0.1188 [0.1177, 0.1200] / 0.1189 [0.1177, 0.1200] |
| ECE, 15 equal-frequency bins (calibrated / raw) | 0.0028 [0.0025, 0.0053] / 0.0041 [0.0032, 0.0063] |
| EBM ROC-AUC / ECE | 0.6892 [0.6853, 0.6930] / 0.0035 [0.0030, 0.0060] |

## Review tiers (calibrated PD anchored on prevalence pi = 0.1483; edges 0.1112 and 0.2224)
| tier | definition | share | observed default rate [95% CI] | mean PD | observed - PD [95% CI] |
|---|---|---|---|---|---|
| Tier 1: lower predicted risk | PD < 0.75 x prevalence | 0.392 | 0.068 [0.066, 0.071] | 0.069 | -0.0008 [-0.0031, +0.0013] |
| Tier 2: typical predicted risk | 0.75 x prevalence <= PD <= 1.5 x prevalence | 0.417 | 0.160 [0.157, 0.164] | 0.162 | -0.0015 [-0.0044, +0.0015] |
| Tier 3: higher predicted risk | PD > 1.5 x prevalence | 0.191 | 0.286 [0.280, 0.292] | 0.282 | +0.0033 [-0.0019, +0.0083] |

## Limitations
- **Research prototype, not for credit decisions;** no fairness, stability or regulatory validation.
- **In-time test split.** Out-of-time (Phase 3) AUC fell by about .016 and calibration drifted: raw ECE .041/.022/.034 and isotonic-on-previous-year ECE .022/.036/.022 for test years 2013/2014/2015 (study Phase 3C), versus an in-time calibrated ECE of 0.0028 for this model, i.e. about 8-13x higher out of time (5-model means from the study; this exact model was not run out of time). **Recalibrate (refit the calibrator on recent labelled loans) at least annually**, and monitor mean PD against the realised default rate.
- **Funded loans only; vintages 2010-2015 only.** Late vintages (2016+) are censored and excluded; the model has not seen 60-month loans issued after 2013.
- **interest_rate is endogenous** (set by the lender from its own risk assessment); removing it costs about .048 AUC and changes capacity-ranking conclusions. Reason codes shown to borrowers hide it by default (config flag).
- Reason-code wording describes the value relative to the training median; it is not causal.
- Inputs outside the accepted ranges are rejected; inputs outside the 0.1%-99.9% training range are scored with a warning.

## Reproduction
Created 2026-10-16T22:19:09+00:00 from commit `f27d7d33bb2d582d7f3a6937ab42871274ae6301` (branch app-v2; working tree dirty at training time: False). Library versions: {"python": "3.13.5", "numpy": "2.4.4", "pandas": "3.0.2", "scikit_learn": "1.8.0", "lightgbm": "4.6.0", "interpret": "0.7.8", "joblib": "1.5.3"}. Schema hash 89c547fad2da2087. Regenerate with `python scripts/train_deploy.py` (about 8 min).
