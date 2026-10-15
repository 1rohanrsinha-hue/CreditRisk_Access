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
- [x] Reproduce baseline Table I first (record env versions; note pandas-3 fill bug effect) — `DONE` — exact after seeding app's unseeded MI step (np.random.seed(42)): LR, CatBoost (AUC .703, thr .519) and RF (rec .668, thr .180) all match Table I

## Phase 2: Core threshold study  (done 2026-10-15)
- [x] Weighting × threshold 2×2 ablation on all 10 models — `DONE` — D2 ×10 repeats, D1 ×5 (`phase2_weighting_*.csv`); RF/ET `min_samples_leaf` sweep ×3 repeats (`phase2_leaf_sweep.csv`)
- [x] Threshold rules: Youden, F1-max, G-mean, fixed-recall (.70/.80), prevalence, Elkan (data-derived costs), profit-max — `DONE` — `creditrisk/thresholds.py`, all picked on OOF, evaluated on test
- [x] Calibration analysis (reliability, Brier, ECE, log loss, Platt/isotonic) — `DONE` — `phase2_calibration_*.csv`, `experiments/figures/reliability_diagrams.png`
- [x] Empirical check: Youden threshold ≈ base rate for calibrated models — `DONE` — H1 SUPPORTED (|t*−π| = .005 D2 / .006–.007 D1)
- [x] Profit/cost evaluation — `DONE` — realized profit = `total_pymnt − funded_amnt` (`total_pymnt` already contains `recoveries`; verified on 100% of Fully Paid/Charged Off rows within $1)
- [ ] Profit evaluation of weighted vs unweighted at equal-data budgets, out-of-time — `TODO` (Phase 3)

### Phase 2 key results (D2, 10 repeats, mean ± sd over repeats)
| cell (all 10 models) | mean recall | recall spread across models | AUC spread |
|---|---|---|---|
| none, thr 0.5 | .058 | .038 ± .005 | .024 |
| none, tuned (Youden OOF) | .669 | .078 ± .026 | .024 |
| balanced, thr 0.5 | .541 | **.633 ± .018** (RF/ET stay at .05) | .027 |
| balanced, tuned | .670 | .065 ± .015 | .027 |

- **RF/ET collapse is not a weighting effect:** balanced RF/ET recall@0.5 = .053/.059 (≈ unweighted). With `min_samples_leaf` 1→5→20 (balanced): ET recall@0.5 .059→.528→.651, Youden t* .203→.434→.482; RF .057→.299→.545, t* .207→.343→.432. Unweighted: t* stays ≈ .20 for every leaf size. ROC-AUC rises .683→.703 (ET) with leaf=20.
- **Calibration (D2):** Platt leaves AUC unchanged (Δ=0.0000), isotonic changes it by −.0005…+.0008 (ties). Raw unweighted Brier .150 / ECE .027; raw balanced Brier .205 / ECE .211; Platt/isotonic all ≈ .147 / .007–.013.
- **Threshold theory:** H1 (calibrated ⇒ t*≈π) SUPPORTED, 40/40 variants within ±.05, mean |t*−π| .005. H2 (balanced raw ⇒ t*≈.5) SUPPORTED 8/10 (fails for RF/ET, t*≈.20: pure leaves). H3 (weighting-invariance after calibration) SUPPORTED: mean |t*_none − t*_bal| = .001–.002 vs .22 raw. AdaBoost raw unweighted t*=.375 (scores not probabilities). Replicated on D1 (H1 .006, H2 .8, H3 .006). Bootstrap sd of t* ≤ .014; ±.005 J-tolerance interval width ≈ .04 (raw .05).
- **Profit (per 1000 applicants, D2 test):** approve-all $402.9k ± 18.5k; oracle $1,853k; reject-all 0. LightGBM isotonic: Youden $425k, F1-max $441k, recall-.80 $322k (loses vs approve-all), Elkan $481k, profit-max $550k (accept 87%, bad rate 16.6%). Paired bootstrap vs Youden: profit-max +$162k [128k, 196k], Elkan +$64k [48k, 81k]; XGBoost F1-max +$0.6k [−2k, +3k] (no difference).
- **Why constant-cost Elkan underperforms profit-max:** repaid-loan profit rises with risk (decile 0→9: $1.1k→$4.9k) and loss-given-default too ($6.3k→$9.4k), so the global costs (C_FP $2,319, C_FN $7,263 ⇒ t=.242) are wrong in the tail; expected profit is negative only in the top risk decile (score > ~.38).

### Limitations of the profit analysis
- Undiscounted cash flows (no time value, no cost of capital).
- Funded loans only: rejected applicants are unobserved (selection bias; approving more than the historical policy is extrapolation).
- No servicing / origination / collection costs. LendingClub `collection_recovery_fee` (mean $202 = 16.7% of gross recoveries on Charged Off) is NOT deducted; would lower charged-off profit by ~2.8%.
- Strict label only (Fully Paid vs Charged Off/Default); Current loans have no realized outcome.
- Charged Off sample is right-censored by vintage (2007–2018 mix); no out-of-time check yet (Phase 3).

## Phase 3: Broader benchmarking
### Phase 3A results (2026-10-15)
- **Strict-labelled rows (full file): 1,345,350** (Fully Paid 1,076,751; Charged Off 268,559; Default 40) of 2,260,668 labelled rows.
- **Vintage audit** (`phase3a_vintage_audit.csv`, `figures/vintage_audit.png`): matured by 2018Q4 = 36m ≤2015, 60m ≤2013. Censored/biased cells: 36m 2016 (28% unresolved), 2017 (60%), 2018 (88%); 60m 2014 (17%), 2015 (33%), 2016–2018 (45–90%). 2007–2008 36m have 35–58% of loans dropped for policy-exception statuses (selection, not censoring). Signature: strict-labelled mean P1 profit falls from +$1.4k (2013, 36m) to −$1.0k (2018) and early-payoff share of Fully Paid → 100%, because performing "Current" loans are excluded (survivorship).
- **Profit definitions:** P1 = `total_pymnt − funded_amnt`; P2(r) = (`total_pymnt` − `collection_recovery_fee`) discounted at r p.a., payments uniform over [issue_d, last_pymnt_d] (or issue_d+term), minus `funded_amnt` at issue.
- **Robustness (D2, 10 rep., LightGBM isotonic, $000/1000 appl.):** approve-all 403 / 363 / −297 / −654 for P1 / P2 0% / 5% / 8%; Youden 425 / 415 / 62 / −129; Elkan 481 / 462 / 65 / −63; profit-max 550 / 527 / 63 / 0. profit-max acceptance: 87% / 86% / 53% / **0%**. Rule ordering is P1/P2-0%: profit-max > Elkan > fixed-0.5 > F1 > Youden > approve-all > recall-.80; at 5% all non-trivial rules tie (Elkan 65, profit-max 63, Youden 62, recall-.80 62; paired CI of profit-max vs Youden includes 0). Recall-≥.80 loses to approve-all under P1/P2-0% (−$97k [−144k, −50k]) but **beats** it by +$339k under 5% because approve-all itself loses money.
- **Expected-profit rule (D2):** P1 $663k vs profit-max $550k vs Elkan $481k (paired +$97k [72k, 122k] and +$195k [165k, 225k]); P2-5%: $169k vs $63k vs $65k (+$118k CI [92k, 139k] vs both). Regressors calibrated (mean predicted vs realized: good $2,324 vs $2,328; bad −$7,206 vs −$7,241; decile table in `phase3a_expprofit_calibration.csv`; R² .63 / .48 on the conditional, .06 on total profit). **On mature cells the rule is below approve-all** (P1 −$32k, P2-5% −$51k), so the D2 gain reflects the censored vintages.
- **OOT (iso, mean of 5 models):** AUC drops .016–.021 vs in-time (RF worst, .644–.654); π: train .142/.154/.124, T−1 .162/.156/.137, test .156/.185/.149; ECE of isotonic-on-T−1 on year T .022/.036/.022 vs in-sample .0025/.0012/.0016 (worse than raw in 2014: .036 vs .023). Approve-all ≥ every threshold rule in all three vintages (P1 $2.08M / 1.37M / 0.84M per 1000 applicants); Youden throws away 57%/36%/28%. **H4 NOT SUPPORTED**: |Δπ| .006/.029/.012 and |Δt*| .009/.014/.009 (corr .33), t* moved *opposite* to π in 2014–2015 (sign agreement 13%), and the Youden-J lost to a stale T−1 threshold is only .0015/.0042/.0025 (0.6–1.6% of J_max, max .007).

- [ ] Repeated stratified CV + bootstrap CIs — `TODO`
- [ ] Friedman/Nemenyi or paired tests — `TODO`
- [ ] PR-AUC, MCC, KS, Gini, balanced accuracy — `TODO`
- [x] Out-of-time validation by `issue_d` — `DONE` (3A): rolling origin T∈{2013,2014,2015}; `oot_study.py`, `phase3a_oot*.csv`
- [ ] Leakage ablation with/without `interest_rate` and engineered `credit_score` — `TODO`
- [ ] Optuna tuning with nested CV — `TODO`
- [ ] SMOTE / undersampling / focal-loss baselines — `TODO`
- [ ] Stacking/voting ensemble — `TODO`
- [ ] WoE scorecard and EBM baselines — `TODO`

## Phase 3B PRE-REGISTRATION (written 2026-10-15, committed BEFORE running tasks C and D; do not edit)
_D3 = strict label; issue years 2010–2013 (any term) + 2014/2015 36-month; population 667,024 loans, default rate .1483; seeded 200k dev sample (prevalence .1477, 91.7% 36-month). The 3A mature-cell diagnostic was run on D2-derived cells (a different sample), so H5/H6 are informed priors, not blind predictions; no D3 outcome has been looked at._

**Headline discount rate: 3% p.a., described only as an illustrative hurdle rate (not claimed to be a cost of funds).** All conclusions are stated across the grid {0, 2, 3, 5, 8}%; P1 (gross, undiscounted) is reported alongside.

| id | Hypothesis | Rationale | Decision rule (fixed now) |
|---|---|---|---|
| H5 | On D3, approve-all earns at least as much as every threshold rule (fixed 0.5, Youden, F1-max, Elkan, profit-max, recall≥.70/.80) at 0% and 3% (and P1); at 8% reject-all or a very high threshold may win | LendingClub prices risk through interest; 3A mature-cell diagnostic | SUPPORTED if no (model, rule≠approve-all) pair at P1/0%/3% has a repeat-mean profit above approve-all together with a paired 95% bootstrap CI (repeat 0) lower bound > 0. REJECTED otherwise (report which) |
| H6 | The expected-profit rule, evaluated out of time (classifier + two LightGBM regressors trained ≤T−2, calibrator and shrinkage margin m chosen on T−1 from {0,50,100,200,300,500,750,1000,1500,2000}$, test T), does not beat approve-all by more than its bootstrap CI | In-time D2 gain was a censoring artefact; mature-cell in-time result was below approve-all | SUPPORTED if in every fold, under P1 and P2-3%, the paired 95% CI of (rule − approve-all) has lower bound ≤ 0. REJECTED if the lower bound is > 0 in ≥2 of 3 folds under both P1 and P2-3% |
| H7 | At matched acceptance rates k∈{30,…,90}% the ten models' profits differ by less than the bootstrap CI half-width at most k, i.e. model family does not matter much for profit, while all models beat random selection at the same k | Phase 2: AUC spread .03, 7 models within .007 | Clause (a): at ≥4 of 7 k levels, max−min of the 10 models' repeat-mean profit < half-width of the 95% bootstrap CI of the best model's profit level. Clause (b): every model has gain over random (profit(k) − k·profit(approve-all)) with paired 95% CI > 0 at every k. H7 SUPPORTED only if (a) and (b) both hold; report each separately |
| H8 | Phase 2 calibration/weighting results replicate on D3 with same direction and similar magnitude | They are properties of probability scales | Replicates if all hold: H1 40/40-style: ≥80% of calibrated variants have \|t*−π\|≤.05; H2: ≥70% of balanced-raw models have \|t*−.5\|≤.05; H3: mean \|t*_none−t*_bal\|≤.01 after calibration; weighting: \|mean tuned recall none − balanced\|≤.02; leaf purity: balanced ET and RF recall@0.5 at leaf 20 exceed leaf 1 by ≥.30 and t* rises by ≥.15 |

Other fixed choices: capacity ranking uses each model's raw (monotone-equivalent) score of the unweighted model to avoid isotonic ties, and a second ranking by expected profit with isotonic-calibrated p; random-selection benchmark = k × approve-all profit; OOT seeds = model seeds 42/43/44, split deterministic; features missing in >50% of the training window are dropped (OOT hygiene).

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
| 2026-10-15 | P2-Q1: ranking metric = realized profit (D2) + ROC-AUC, PR-AUC, Brier; never recall. F1/MCC/recall/precision secondary at the chosen point | Recall spread is threshold noise |
| 2026-10-15 | P2-Q2: uncertainty everywhere: 10 repeated stratified 64/16/20 splits on D2, 5 on D1 (split seed 42+rep, model seed 42+rep, data sample fixed); 1000-resample bootstraps (t* CI on rep 0, paired bootstrap for profit diffs) | ±.03 recall noise |
| 2026-10-15 | P2-Q3: weighting 2×2 = {none, balanced} × {0.5, tuned (Youden on OOF)}; balanced = class_weight / `sample_weight` for GB, AdaBoost, **and MLP** (sklearn 1.8 supports it, no oversampling needed) / XGB `scale_pos_weight`=neg/pos of each training fold / CatBoost `auto_class_weights` | As specified |
| 2026-10-15 | P2-Q4: calibration protocol: 5-fold OOF (fresh pipeline per fold) over train+val → Platt (LR on logit) and isotonic fit on OOF → all thresholds on calibrated OOF → refit on train+val → apply calibrators → test | As specified; D2 full, D1 replication |
| 2026-10-15 | P2-Q5: realized_profit = total_pymnt − funded_amnt; `total_pymnt` = principal+interest+late fee+recoveries holds for 100% of FP/CO rows (32% without recoveries), so adding `recoveries` would double count. Elkan costs derived from train+val data | Verified on 2.26M rows |
| 2026-10-15 | P2-Q6: runtime estimate ~95 min sequential-CPU (~45 min actual wall clock with 3+1+1 workers); full repeats kept (no cuts) | Under budget |
| 2026-10-15 | P2-Q7: committed app.py (tag `paper-v1-baseline`) and .gitignore as separate commits on journal-extension; main untouched; nothing pushed; venv.txt untracked | As requested |
| 2026-10-15 | P3B-Q1: re-base on D3 (mature vintages); D2 = sensitivity, D1 = replication; core Phase 2 results re-run on D3 | Headline numbers must not come from a censored sample |
| 2026-10-15 | P3B-Q2: profit message reframed ("threshold rules do not beat risk-priced lending on mature vintages; thresholds matter under a capacity constraint"); with/without interest_rate comparison belongs to 3C | |
| 2026-10-15 | P3B-Q3: headline discount 3% p.a. (illustrative hurdle only), grid {0,2,3,5,8}% | |
| 2026-10-15 | P3B-Q4: 3C inference = paired bootstrap per fold + Holm; equivalence margins (AUC ±.005) rather than non-rejection; Friedman/Nemenyi only secondary | ~5 blocks = low power |
| 2026-10-15 | P3B-Q5: Optuna tunes log loss only; profit reported, never tuned (3D) | Profit is noisy under censoring |
| 2026-10-15 | P3B-Q6: OOT: 3 model seeds per fold; split deterministic | |
| 2026-10-15 | P3A-Q1: rolling-origin OOT on issue year, T∈{2013,2014,2015}: fit ≤T−2, calibrate+thresholds on T−1, test T; all terms for 2013/14, 36m only for 2015; vintages ≥2016 dropped from headline; windows >400k subsampled (seeded; none triggered) | As specified |
| 2026-10-15 | P3A-Q2: report P1 and P2 (5%, with 0%/8% sensitivity, net of `collection_recovery_fee`) | Totals only, no cash-flow timing |
| 2026-10-15 | P3A-Q3: headline decision metric NOT chosen yet; report profit-max (OOF), data-derived Elkan and the expected-profit rule side by side under P1/P2 | Decision after 3A |
| 2026-10-15 | P3A-Q4: score-dependent expected-profit baseline added (`creditrisk/expected_profit.py`) | As specified |
| 2026-10-15 | P3A-Q5/Q6: stats (Nadeau-Bengio, paired bootstrap + Holm; blocks = vintage folds + datasets) and Optuna plan (25 trials, inner 3-fold, outer 5-fold, 5 models, log-loss, 100k subsample, ≤3h) deferred to 3B/3C | As specified |
| 2026-10-15 | P3A-Q7: claim table replaced by the author's manuscript mapping | |
| 2026-10-15 | Bug fix: `FrameEncoder` median of an all-NaN column (e.g. `mort_acc` before 2012) now 0 instead of NaN; no effect on D1/D2 results | Needed for OOT |
| 2026-10-15 | **Finding:** strict-labelled 2016–2018 loans are survivorship-biased (performing Current loans excluded), so Phase 2/3A-C D2 profit levels and rule rankings are partly a censoring artefact; the headline profit analysis should move to mature vintage cells | `mature_subset_check.py` |
| 2026-10-15 | Seeding: baseline_repro seeds numpy (42) before app's unseeded `mutual_info_classif`; creditrisk code has no unseeded RNG. **Result: Table I reproduces exactly** (LR .592/.269/.718/.391/.696/.452; CatBoost AUC .703 thr .519; RF rec .668 thr .180). The earlier unseeded run differed by up to .038 recall / .012 threshold (column-order noise, `rf_column_order_sensitivity.csv`) | As requested |
| 2026-10-14 | Protocols A/B/C (see results): leaking the threshold to the test set changes mean F1 by ≤ .003 and mean recall by ≤ .03; per-model recall moves ±.02–.04 | `protocol_comparison_all.csv` |

## Surprising findings (Phase 1)
- Youden-threshold recall for a single model is unstable to *column order* (RF std .032) — larger than most inter-model recall differences claimed in the paper.
- Model ranking by recall is essentially uncorrelated across threshold protocols (Spearman A~B ≈ -0.4/0.0, A~C ≈ 0.15/0.05 for D1/D2).
- Test-set threshold tuning (protocol A) inflates aggregate metrics only slightly; the bigger issue is variance of the Youden point on a flat ROC.
- At fixed 0.5, recall is ~0.02–0.06 for GB/AdaBoost/MLP/ET/RF but ~0.62–0.67 for the others: that cluster split is a probability-scale (weighting/bagging) effect, not ranking quality.
- `sel_stability` = 1.0 trivially since `select_k=None` (all features kept).

## Paper claims affected
_Mapped to the manuscript sections by the author (2026-10-15)._
| Manuscript location | Claim | Status | Evidence |
|---|---|---|---|
| Abstract / Sec V | "Threshold matters more than model family" | **CONFIRMED** for 0.5-vs-tuned; **WEAKENED** for tuned-recall ranking | recall spread at 0.5 = .63 vs AUC spread .027; after tuning spread .065–.078 is threshold noise (rank ρ≈0 across protocols, RF sd .032) |
| Sec IV-B | "XGBoost and LightGBM offer the most desirable recall–AUC balance" | **WEAKENED** | ranks flip across protocols/seeds; 7 of 10 models within .007 AUC |
| Sec III-F | "Conservative probabilities rarely exceed 0.5" | **PARTIAL** | true for unweighted GB/AdaBoost/MLP where t*≈base rate; RF/ET mechanism is leaf purity (min_samples_leaf 20 ⇒ ET recall@0.5 .651) |
| Sec III-F | "Youden gives the best operating point" | **CHANGED** | not profit-optimal; and in mature vintages no cutoff beats approve-all (Phase 3A) |
| Sec III-B2 | Imputation paragraph | **WRONG** | pandas fill bug: nothing was imputed (6151 NaNs zero-filled); journal version uses in-fold imputation |
| Sec V | "CatBoost native categorical handling" | **WRONG** | `cat_features` was never passed in the paper's code; fixed in `creditrisk/models.py` |

### Phase 3A changes to the picture
- The Phase 2 profit conclusions (profit-max accepts ~87%, +$162k over Youden, recall-.80 loses to approve-all) are **conditional on a D2 sample that is ~50% censored vintages**. On mature cells only (≤2013, or 2014/2015 36-month) approve-all is optimal, Youden loses 53%, and even the expected-profit rule is −$32k/−$51k below approve-all.
- Under 5% p.a. discounting the whole book is unprofitable (approve-all −$297 per applicant on D2; +$475/applicant on mature cells), and at 8% profit-max rejects everything on D2.

### Limitations added in Phase 3A
- **Hindsight labels:** the T−1 calibration/threshold set uses outcomes that were only observable after year T−1 (a loan issued in 2012 resolves in 2015); a live system would have fewer resolved labels.
- **Censoring:** strict labelling drops unresolved loans; late vintages are survivorship-biased and 60-month 2014–2015 cells are truncated. Headline results restricted to mature cells / vintages ≤2015.
- **Discounting approximation:** only totals are available; payments are assumed uniform between issue and last payment (recoveries on charged-off loans may arrive later). 5%/8% rates are scenario hurdles, not a funding-cost estimate.
- **Selection:** funded loans only; "approve-all optimal" means "within the historical approved population", not for rejected applicants.

## Open questions
- [ ] Which environment produced the paper's Table I (pandas version)? The inplace-fillna bug is pandas-3 specific.
- [ ] Target journal / page budget / required reporting standards?
- [ ] Full 2.26M rows, or a stratified/time-based subsample (compute budget)?
- [ ] Label policy: drop `Current` and `Late/Grace` loans, or keep current definition as the headline?
- [ ] Is `interest_rate`/grade-derived `credit_score` allowed as a feature in the headline results?
- [ ] Which python env to standardise on (rebuild `venv`; remove Windows `.venv`)?
- [ ] Uncommitted changes to `app.py`/`.gitignore` on main: intentional? Commit them before Phase 1?
- [ ] (Phase 3) Which temporal split for out-of-time: train ≤2015 / test 2016–2018? Current loans in late vintages are censored — drop vintages after 2016?
- [ ] (Phase 3) Deduct `collection_recovery_fee` from realized profit (−2.8% on charged-off loss)? Default: no, per instruction.
- [ ] (Phase 3) Optuna budget and the nested-CV outer folds (compute)?
- [ ] (Phase 3) Is a time-value discount (e.g. 5–8% p.a. on monthly cash flows) acceptable, given only totals are available per loan?
