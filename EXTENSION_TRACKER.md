# Journal Extension Tracker

Status legend: `TODO` / `IN PROGRESS` / `DONE`. Last updated: 2026-10-14.

## Overview
- [ ] **Paper title:** _TBD (conference paper title to be filled in by author)_ — `TODO`
- [ ] **Goal:** expand the conference paper into a journal paper (threshold/weighting study on Lending Club credit-risk models) — `IN PROGRESS`
- [ ] **Baseline:** current `app.py` results (Table I: 10 models, Youden threshold, metrics on 20% test split of first 100k rows) — frozen as reference; must be reproduced first — `TODO`

## Audit findings (app.py, verified 2026-10-14)
- [x] 1. Youden threshold computed on `y_test` and metrics reported on same `y_test` — **CONFIRMED** (app.py:771-785) — `DONE` in `creditrisk/` — new pipeline selects threshold on val/OOF (protocol B/C) (app.py unchanged)
- [x] 2. Imputation, scaler and feature selection fit on full data before split — **CONFIRMED** (app.py:725, 731-742; medians at 252) — `DONE` in `creditrisk/` — all preprocessing fit on train only (FrameEncoder in sklearn Pipeline) (app.py unchanged)
- [x] 3. `nrows=100000` = first 100k rows, not random — **CONFIRMED** (app.py:160,163); issue_d is only Oct–Dec 2015 (single quarter) — `DONE` in `creditrisk/` — seeded random sample of labelled rows from cached full file; `--sample-size` everywhere (app.py unchanged)
- [x] 4. Feature selection is a no-op (16 in, 16 out; only reorders) — **CONFIRMED** (app.py:322-361, 725) — `DONE` in `creditrisk/` — selection is real + optional (`select_k`, in-fold); default keeps all (app.py unchanged)
- [x] 5. CatBoost gets label-encoded categoricals, no `cat_features`; employment_length encoded alphabetically — **CONFIRMED** (app.py:717-722, 383) — `DONE` in `creditrisk/` — CatBoost gets `cat_features`, LightGBM pandas-category, ordinal emp_length (app.py unchanged)
- [x] 6. GradientBoosting, AdaBoost, MLP unweighted; others weighted — **CONFIRMED** (app.py:374, 385, 387); XGB uses hardcoded `scale_pos_weight=4.48` (377) — `IN PROGRESS` — weights now computed from train labels; legacy weighting scheme kept for the Phase 2 ablation
- [x] 7. SHAP explainer only for best model; global plots use 50 samples — **CONFIRMED** (app.py:833-842, 587); also `shap` not installed in either env — `TODO`
- [x] 8. `/predict` and `/api/predict` use 0.5 cutoff + hardcoded bands on uncalibrated probs, not Youden — **CONFIRMED** (app.py:1049-1068, 1097, 1154-1160) — `TODO`
- [x] 9. `models/` never written/loaded; `joblib` imported but unused — **CONFIRMED** (only `import joblib` at app.py:73); `models/*.pkl` are stale, git-ignored artifacts — `TODO`
- [x] 10. Missing employment_length: mode-fill intended, but **in practice zero-fill** under pandas 3.0.2 — **CONFIRMED** (app.py:253, 261 `fillna(inplace=True)` on a column is a no-op under Copy-on-Write; fallback `fillna(0)` at 266 fires; verified 6151 NaNs left, 6112 of them emp_length → label "0"; `'n/a'→'< 1 year'` map at 212 never hit because missing is NaN) — `DONE` in `creditrisk/` — new pipeline: median/mode in-fold + emp_length missing indicator; legacy bug replicated only behind `legacy_fill_bug` (app.py unchanged)

### Additional findings (not in original list)
- [ ] Same inplace-fillna no-op zero-fills numeric NaNs (dti 2, revol_util 37) — `TODO`
- [ ] Label: `Current` (11.4% of rows) counted as non-default (right-censored); `Late (16-30)` counted non-default, `Late (31-120)`/`In Grace Period` counted default — `TODO`
- [ ] `credit_score` is synthetic: derived from `grade` and `interest_rate` (app.py:211-216) — leakage/redundancy; user-entered in /predict — `TODO`
- [ ] Best model chosen by test ROC-AUC (selection on test) (app.py:834, 931) — `TODO`
- [ ] `cross_val_score` run on already-scaled train data and doubles runtime; CV scaler leak is minor — `TODO`
- [ ] Training runs inside the request path on first hit (`/dashboard`, etc.) if not started via `__main__` — `TODO`
- [ ] `requirements.txt` includes `shap==0.51.0` but neither `venv` nor `.venv` has shap installed (app prints "SHAP not available") — `TODO`
- [ ] `.venv` is a Windows venv (Python 3.14, `Scripts/`, another user's path) — unusable on this Mac; `venv` is the real env (Python 3.13.5) but is incomplete; `venv.txt` has a stale path — `TODO`
- [ ] Working tree was not clean at start: `.gitignore` and `app.py` have uncommitted modifications, `venv.txt` untracked (not touched) — `TODO`

## Phase 1: Methodological hardening
- [x] Leak-free train/val/test split (fit all preprocessing on train only) — `DONE` — `creditrisk/splits.py`, stratified 64/16/20, seed from `Config`
- [x] Threshold tuned on validation or OOF predictions, never test — `DONE` — `experiments/protocol_comparison.py` (protocols A/B/C); headline pipeline uses B/C from now on
- [x] sklearn `Pipeline`/`ColumnTransformer` for impute/encode/scale — `DONE` — custom `FrameEncoder` (impute/encode/scale) inside sklearn `Pipeline`; fit on train folds only
- [x] Random or full-data loading (replace first-100k) — `DONE` — parquet cache `.cache/lc_full.parquet` (2.26M rows); seeded 200k dev sample; first-100k only for legacy
- [ ] Label-definition sensitivity (Fully Paid vs Charged Off only) — `IN PROGRESS` — both labels implemented (`label_legacy`/`label_strict`); D1→D2 changes data+label+features at once, single-factor ablation still to run
- [x] Ordinal `employment_length` (0–10, proper NaN handling) — `DONE` — 0..10 + explicit missing indicator
- [x] CatBoost native categoricals (`cat_features`) — `DONE` — `cat_features` set; also LightGBM category dtype
- [x] Add `fico_range_low/high` features — `DONE` — `fico_mean` replaces grade-derived `credit_score` in headline features
- [x] Seeds + config file (single source of truth) — `DONE` — `creditrisk/config.py` `Config` dataclass (seed 42, split sizes, leakage switches); no YAML yet
- [x] Reproduce baseline Table I first (record env versions; note pandas-3 fill bug effect) — `DONE` — LR + CatBoost AUC exact; RF recall within column-order noise (see Decisions log)

## Phase 2: Core threshold study
- [ ] Weighting × threshold 2×2 ablation on all 10 models — `TODO`
- [ ] Threshold rules: Youden, F1-max, G-mean, fixed-recall, prevalence, cost-sensitive — `TODO`
- [ ] Calibration analysis (reliability, Brier, ECE, Platt/isotonic) — `TODO`
- [ ] Empirical check: Youden threshold ≈ base rate for calibrated models — `TODO`
- [ ] Profit/cost evaluation using `funded_amnt`, `total_pymnt`, `recoveries` — `TODO`

## Phase 3: Broader benchmarking
- [ ] Repeated stratified CV + bootstrap CIs — `TODO`
- [ ] Friedman/Nemenyi or paired tests — `TODO`
- [ ] PR-AUC, MCC, KS, Gini, balanced accuracy — `TODO`
- [ ] Out-of-time validation by `issue_d` — `TODO`
- [ ] Leakage ablation with/without `interest_rate` and engineered `credit_score` — `TODO`
- [ ] Optuna tuning with nested CV — `TODO`
- [ ] SMOTE / undersampling / focal-loss baselines — `TODO`
- [ ] Stacking/voting ensemble — `TODO`
- [ ] WoE scorecard and EBM baselines — `TODO`

## Phase 4: Explainability
- [ ] SHAP on all models with 2000+ samples — `TODO`
- [ ] Rank correlation of attributions across models — `TODO`
- [ ] Top-k reason codes — `TODO`

## Phase 5: Deployment
- [ ] Persist models/scaler/encoders with joblib (versioned) — `TODO`
- [ ] Calibrated PD output — `TODO`
- [ ] Risk bands and cutoff derived from Phase 2 results — `TODO`
- [ ] Fix `/api/predict` (encoder handling, defaults, per-model threshold) — `TODO`
- [ ] Docker — `TODO`
- [ ] Basic tests — `TODO`

## Phase 6: Paper support
- [ ] Auto-export results to CSV and LaTeX tables — `TODO`
- [ ] 300 dpi figures — `TODO`
- [ ] List of paper claims that changed vs. conference version — `TODO`

## Decisions log
| Date | Decision | Reason |
|------|----------|--------|
| 2026-10-14 | Work on branch `journal-extension`; do not modify `app.py`, templates, or the CSV | Keep baseline reproducible; new code goes in new files |
| 2026-10-14 | Use `/Library/Developer/CommandLineTools/usr/bin/git` (default `git` blocked by Xcode license prompt) | Avoid needing sudo |
| 2026-10-14 | Q1: treat pandas 3.0.2 in `venv/` as the baseline env; legacy fill bug replicated only via `legacy_fill_bug=True`; also run properly-imputed variant; never inplace-fillna on a selection in new code | Table I was produced under the bug |
| 2026-10-14 | Q2: full CSV cached once to `.cache/lc_full.parquet` (gitignored; `.gitignore` edited by append only, not committed); dev data = seeded 200k sample, drawn from *labelled* rows (after label filter) so D2 has exactly 200k modelled rows; full 2.26M for out-of-time/final runs | Compute + reproducibility |
| 2026-10-14 | Q3: headline label = Fully Paid vs Charged Off/Default; legacy 5-status label only for reproduction + sensitivity; report both | Remove right-censored/noisy labels |
| 2026-10-14 | Q4: headline features = application-time incl. `interest_rate`, with `fico_mean` instead of grade-derived `credit_score`; leakage switches (`use_interest_rate`, `credit_score` in {fico, grade_derived, none}) implemented, ablation run in Phase 3 | Avoid grade/rate leakage in the headline |
| 2026-10-14 | Q5: installed shap, pyarrow, optuna, interpret, imbalanced-learn, statsmodels, scikit-posthocs into `venv/`; `requirements-lock.txt` from pip freeze; `requirements.txt` untouched; Windows `.venv` ignored | Co-author's env |
| 2026-10-14 | Q6: app.py/.gitignore uncommitted edits left alone. The app.py diff *is* the Youden-threshold code behind Table I (HEAD still predicts at 0.5) | Table I can only come from the working tree |
| 2026-10-14 | Q7: layout `creditrisk/` (package), `experiments/` (scripts), `experiments/results/` (small CSVs, force-added because `*.csv` is gitignored) | As requested |
| 2026-10-14 | Table I repro: LR matches to 3 d.p. (Acc .592, Prec .269, Rec .718, F1 .391, AUC .696, Thr .452); CatBoost AUC .703 exact, Thr .510 vs .519; RF Rec .641 vs .668, Thr .190 vs .180. RF/CatBoost differences are from column order: app.py's feature selection reorders columns using unseeded `mutual_info_classif`; over 20 random column orders RF recall = .669 ± .032 (range .600–.706), thr .1805 ± .010 — the paper's .668 is the mean. Not a replica error. | `experiments/column_order_sensitivity.py` |
| 2026-10-14 | Fill-bug effect (proper vs bug): AUC changes ≤ .002 for all models; tuned-threshold recall/precision change up to ±.04–.06 for RF/CatBoost/ET/LGBM (same column-order noise scale) | `baseline_table1.csv` |
| 2026-10-14 | Protocols A/B/C (see results): leaking the threshold to the test set changes mean F1 by ≤ .003 and mean recall by ≤ .03; per-model recall moves ±.02–.04 | `protocol_comparison_all.csv` |

## Surprising findings (Phase 1)
- Youden-threshold recall for a single model is unstable to *column order* (RF std .032) — larger than most inter-model recall differences claimed in the paper.
- Model ranking by recall is essentially uncorrelated across threshold protocols (Spearman A~B ≈ -0.4/0.0, A~C ≈ 0.15/0.05 for D1/D2).
- Test-set threshold tuning (protocol A) inflates aggregate metrics only slightly; the bigger issue is variance of the Youden point on a flat ROC.
- At fixed 0.5, recall is ~0.02–0.06 for GB/AdaBoost/MLP/ET/RF but ~0.62–0.67 for the others: that cluster split is a probability-scale (weighting/bagging) effect, not ranking quality.
- `sel_stability` = 1.0 trivially since `select_k=None` (all features kept).

## Open questions
- [ ] Which environment produced the paper's Table I (pandas version)? The inplace-fillna bug is pandas-3 specific.
- [ ] Target journal / page budget / required reporting standards?
- [ ] Full 2.26M rows, or a stratified/time-based subsample (compute budget)?
- [ ] Label policy: drop `Current` and `Late/Grace` loans, or keep current definition as the headline?
- [ ] Is `interest_rate`/grade-derived `credit_score` allowed as a feature in the headline results?
- [ ] Which python env to standardise on (rebuild `venv`; remove Windows `.venv`)?
- [ ] Uncommitted changes to `app.py`/`.gitignore` on main: intentional? Commit them before Phase 1?
