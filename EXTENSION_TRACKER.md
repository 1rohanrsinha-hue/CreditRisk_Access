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
- [ ] `credit_score` is synthetic: derived from `grade` and `interest_rate` (app.py:211-216) — endogenous/redundant feature (see terminology note); user-entered in /predict — `TODO`
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
- [x] Seeds + config file (single source of truth) — `DONE` — `creditrisk/config.py` `Config` dataclass (seed 42, split sizes, endogenous-feature switches); no YAML yet
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
- **OOT (iso, mean of 5 models):** AUC drops .016–.021 vs in-time *[Phase 6 check against the 3-seed CSV: fold means .0154/.0209/.0176, i.e. .015–.021]* (RF worst, .644–.654); π: train .142/.154/.124, T−1 .162/.156/.137, test .156/.185/.149; ECE of isotonic-on-T−1 on year T .022/.036/.022 vs in-sample .0025/.0012/.0016 (worse than raw in 2014: .036 vs .023). Approve-all ≥ every threshold rule in all three vintages (P1 $2.08M / 1.37M / 0.84M per 1000 applicants); Youden throws away 57%/36%/28%. **H4 NOT SUPPORTED**: |Δπ| .006/.029/.012 and |Δt*| .009/.014/.009 (corr .33), t* moved *opposite* to π in 2014–2015 (sign agreement 13%), and the Youden-J lost to a stale T−1 threshold is only .0015/.0042/.0025 (0.6–1.6% of J_max, max .007).

- [ ] Repeated stratified CV + bootstrap CIs — `TODO`
- [ ] Friedman/Nemenyi or paired tests — `TODO`
- [ ] PR-AUC, MCC, KS, Gini, balanced accuracy — `TODO`
- [x] Out-of-time validation by `issue_d` — `DONE` (3A): rolling origin T∈{2013,2014,2015}; `oot_study.py`, `phase3a_oot*.csv`
- [x] Endogenous-feature ablation with/without `interest_rate` and engineered `credit_score` — `DONE` (3C; `experiments/endogenous_feature_ablation.py`, `phase3c_ablation*.csv`)
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

## Phase 3C PRE-REGISTRATION (written 2026-10-16, committed BEFORE tasks B–E; H5–H8 at 4f86f74 remain frozen)
_Informed priors: 3B showed default-risk ranking earns less than random selection by count-capacity on D3, and that `interest_rate` x loan size drive profit regressions. Nothing below has been evaluated on the 3C analyses (stats, dollar-capacity/ROI, ablation cells, drift fixes)._

| id | Hypothesis | Rationale | Operationalisation / decision rule (fixed now) |
|---|---|---|---|
| H9 | On D3, a majority of the boosting family (GB, XGBoost, LightGBM, CatBoost, AdaBoost) is equivalent to the best model on ROC-AUC | Phase 2/3B: 7 of 10 models within .007 AUC | TOST at ±0.005 AUC, Nadeau–Bengio corrected (variance factor 1/J + n_test/n_train, J=10, n_te/n_tr=.25), Holm over the 9 comparisons per dataset. SUPPORTED if ≥3 of the 5 boosting models (the best model counts itself if it is boosting) are equivalent to the best model at Holm-adjusted α=.05. Models outside the band are reported |
| H10 | H1 holds in every leakage-ablation cell | The t*=π theory does not depend on features | For every (cell, model, calibrator∈{Platt, isotonic}) the mean-over-repeats \|t*−π\| < 0.02. SUPPORTED only if all cells pass |
| H11 | Removing interest_rate and grade-derived score (cell F vs cell A) lowers ROC-AUC by > .005 and makes the default-risk-ranked capacity gain over random strictly larger (less negative) | interest_rate is the lender's risk price; low-risk loans earn less interest | Both must hold, as the mean over 5 models and 5 repeats on D3: ΔAUC(A−F) > .005 and gain_F > gain_A, with gain = profit per 1000 applicants − k × approve-all profit, dollar capacity, k=50%, P2-3%, default-risk (raw-score) ranking. Direction only |
| H12 | Under dollar-capacity the expected-profit ranking's gain over random is smaller than under count-capacity (loan-size confound) | Expected-profit ranking may favour large loans | Mean over k∈{30..90}% and the 10 models of (gain_dollar − gain_count) < 0 for the expected-profit ranking at P2-3%, with the paired-by-repeat 95% CI (10 repeats, t-interval) excluding 0 |
| H13 | With interest_rate in the features, default-risk ranking still earns less than random selection under ROI at k=50%, and an interest_rate-only ranking is not distinguishable from the best model ranking on ROI | Pricing drives return | (a) dollar capacity, k=50%, P2-3%: for each of the 10 models the repeat-mean ROI of the default-risk ranking < ROI(approve-all) (random selection has the same expected ROI). (b) the ranking by interest_rate alone, **higher rate first** (the ROI-maximising direction; lowest-first is reported as a baseline), vs the best model-based ranking (best over 10 models × {default-risk, expected-profit, expected-ROI} by repeat-mean ROI): paired bootstrap 95% CI (repeat 0, 1000 resamples) of ΔROI lies within ±0.02 × ROI(approve-all). H13 SUPPORTED only if (a) and (b) hold |
| H14 | At least one cheap recalibration fix reduces the OOT ECE of isotonic-on-T−1 by ≥30% in ≥2 of 3 folds | Drift in π and score scale | Fixes: Platt on T−1; isotonic on trailing 6 months (Jul–Dec) of T−1; isotonic + prior-shift correction with Saerens-style EM estimate of π_T from unlabeled test scores. ECE averaged over 5 models × 3 seeds per fold; reduction vs the isotonic-on-T−1 baseline. The oracle (true π_T) is reported but excluded from H14. All fixes reported regardless |

Other fixed choices: leakage cells A (rate in, fico = headline), B (in, none), C (in, grade-derived), D (out, fico), E (out, grade-derived; a *partial* removal because the grade-derived score embeds the rate), F (out, none); D3, 5 repeats, models LR/RF/XGBoost/LightGBM/CatBoost, unweighted; OOT 10 models seed 42 (3 seeds for the five main models); ROI = realized profit / funded_amnt; bad rate reported count- and dollar-weighted.

## Phase 3B results (2026-10-16; D3 = mature vintages; outcomes reported against the frozen pre-registration above)
**Datasets.** D3 population 667,024 strict-labelled loans (default rate .1483; 2010–2013 any term + 2014/2015 36m; 91.7% 36-month in the 200k sample, prevalence .1477). D1 100k legacy-label/one quarter (π .182); D2 200k strict, 2007–2018 incl. 38.9% from ≥2016 (π .199); D3 π .148.

| claim | D1 | D2 | D3 | verdict |
|---|---|---|---|---|
| recall@0.5 unweighted → balanced (mean of 10 models) | .039 → .534 | .058 → .541 | .005 → .522 | SURVIVES |
| tuned-recall \|none − balanced\| | .005 | .001 | .005 | SURVIVES |
| tuned-recall spread across models (balanced) | .081 | .065 | .077 | SURVIVES (threshold noise) |
| ROC-AUC spread across models | .036 | .027 | .032 | SURVIVES |
| H1 calibrated t*≈π: share within ±.05 / mean \|t*−π\| | 100% / .006 | 100% / .005 | 100% / .004 | SURVIVES |
| H2 balanced raw t*≈.5: share within ±.05 (RF/ET fail) | 80% | 80% | 80% | SURVIVES |
| H3 weighting gap after calibration | .004 | .002 | .002 | SURVIVES |
| raw balanced ECE → calibrated ECE | .224 → .012 | .211 → .010 | .254 → .008 | SURVIVES |
| leaf purity: balanced ET recall@.5 leaf 1→20; t* | n/a | .059→.651; .203→.482 | .006→.631; .147→.495 | SURVIVES |
| leaf purity: balanced RF recall@.5 leaf 1→20; t* | n/a | .057→.545; .207→.432 | .003→.451; .147→.416 | SURVIVES |

**H8: REPLICATES** — all six pre-registered criteria pass on D3 (`phase3b_replication.csv`). Platt AUC change 0.0000, isotonic −.0006…+.0012.

**H5 (in-time D3, 5 models × 10 repeats): SUPPORTED at P1, P2-0%, P2-3%** — 0 (model, rule) pairs with repeat-mean above approve-all and paired CI lower bound > 0. LightGBM, $000 per 1000 applicants, 3%: approve-all 763 (sd 10) | fixed-0.5 763 | profit-max 763 (accepts 99.9%) | expected-profit 766 | Elkan 675 | F1-max 474 | Youden 410 | recall ≥.70 342 | recall ≥.80 249. Paired CI vs approve-all (rep 0, 3%): Youden −$343k [−374k, −312k], recall-.80 −$493k, Elkan −$80k, expected-profit −$5k [n.s.].
- Profit-max acceptance by discount rate (LightGBM): P1 100%, 0% 100%, 2% 99.9%, 3% 99.9%, 5% 99.8%, 8% 95.5%.
- **Cells where a rule beats approve-all with paired CI excluding 0 (in-time D3):** none at P1/0%/2%/3%/5%. At **8%** (the pre-registered exception): expected-profit beats approve-all for LR/LightGBM/XGBoost/CatBoost/GB by +$110k…+$126k per 1000 (LightGBM +$115k [96k, 134k]; accepts 47%) and profit-max for CatBoost by +$7.1k [2.2k, 12.3k]. At 8% approve-all itself earns only $59k per 1000.
- **Separate OOT result (not part of H5):** 2015 36-month fold, P2-3%, Elkan with thresholds from T−1 earned +$7.7k per 1000 over approve-all, paired CI [3.3k, 11.9k] (identical for seeds 42/43/44 — the LightGBM fit has no seed-dependent subsampling here; accepts 92.5%). This is an out-of-time, single-fold, single-vintage result; H5 is evaluated on the in-time D3 analysis only.

**H6 (OOT expected-profit rule, 3 folds × 3 seeds): SUPPORTED.** Lower bound of the paired CI vs approve-all is ≤ 0 in every fold/scenario/seed (P1: −14.9k/−1.7k/−1.2k; P2-3%: −32.2k/−3.1k/−0.6k, seed 42); folds with lower bound > 0: none. Mean differences: P1 −12.6k/−0.8k/−0.05k; P2-3% −28.6k/−1.5k/+1.5k (n.s.). The shrinkage margin chosen on T−1 was m=0 in all folds. Youden(T−1) loses $216k–$436k per 1000 in 2014/2015; Elkan(T−1) loses $186k in 2013.

**H7 (capacity, 10 models, 10 repeats + 3 OOT folds): NOT SUPPORTED** (pre-registered primary ranking = default-risk score).
- Clause (a): range of model profits < CI half-width at only 3/7 acceptance levels (needs ≥ 4); clause (b): **0/7** — selecting the k% lowest-risk applicants earns *less than random selection* for all 10 models at k=30–70% (gain over random −$21k…−$53k per 1000 at P2-3%; CI upper bound < 0 for 10/10 models at k ≤ .7, 9/10 at .8, 2/10 at .9) even though bad rate falls from .148 to .063–.130, because low-risk loans carry lower interest.
- Secondary ranking (expected profit, isotonic p + LightGBM regressors): both clauses hold — range $19–30k < CI half-width $32–35k at 7/7 levels; all 10 models beat random at every k (+$66k…+$266k per 1000, CI lower bound > 0). At k=90% profit $753k vs approve-all $763k.
- Model-rank stability is low: mean Spearman across k .58 (P2-3%) / .61 (P1); across 10 repeats at k=.5 .46 / .40; across OOT folds .17 / .16.

**OOT hygiene and seeds.** Features missing in >50% of the training window are dropped (`mortgage_accounts` for T=2013, 2014; none for 2015). All 5 models × 3 folds changed by ≤ .0014 AUC vs 3A (threshold .002) so no full rerun was needed. 3 model seeds: seed sd of OOT AUC ≈ .0003 (RF .0007–.0011; LR/LightGBM/XGBoost deterministic ≈ 0). Fold means (5 models × 3 seeds): AUC in-time .682/.697/.691 → OOT .666/.676/.673; ECE of isotonic-on-T−1 on T .022/.036/.022 vs in-sample .0026/.0010/.0016; J lost to stale T−1 threshold .0021/.0040/.0035. P2-3% profit per 1000 (mean of models ± seed sd): approve-all 1,453 / 803 / 394; Youden 601±16 / 552±5 / 318±1; Elkan 1,255±5 / 803±0.2 / 402±0.2; profit-max 1,444±2 / 803 / 395.

**Operational notes.** The laptop did not actually sleep (uptime 16 days); the background jobs ran to completion. Score and analysis runners skip existing units (atomic `os.replace` checkpoints). `experiments/status.sh` prints progress.

## Phase 3D PRE-REGISTRATION (written 2026-10-16, committed BEFORE tasks B–F; H5–H14 remain frozen)
_Wording rule: "not equivalent" is not "worse". If a TOST fails but the difference CI includes 0 the cell is labelled INCONCLUSIVE. All equivalence tests: ROC-AUC, margin ±0.005, Nadeau–Bengio corrected (variance factor 1/J + n_test/n_train, n_te/n_tr = .25), Holm over the stated family._

| id | Hypothesis | Rationale | Decision rule (fixed now) |
|---|---|---|---|
| H15 | Random undersampling and SMOTE (inside training folds only; OOF validation folds and calibration data never resampled) move the raw Youden t* toward 0.5 like balanced weighting, calibrated t* stays within .02 of π, and AUC is equivalent (±.005) to the unweighted baseline | Resampling, like weighting, only shifts the effective prior | D3, 5 repeats, LR/RF/LightGBM/XGBoost. (i) for each method, \|t*_method − .5\| < \|t*_none − .5\| (raw Youden on OOF) in ≥3 of 4 models; (ii) every (method, model, calibrator∈{Platt, isotonic}) mean \|t*−π\| < .02; (iii) TOST equivalence vs unweighted for every (method, model), Holm over 8. SUPPORTED only if (i)–(iii) all hold; each reported separately |
| H16 | After nested-CV tuning: (a) AUC gain of tuned over default < .005 for LightGBM, XGBoost, CatBoost; (b) RF gains more than the boosters; (c) AUC spread across the 5 tuned models ≤ spread at defaults | Phase 2/3C: GBDTs sit on a plateau; RF is limited by leaf purity | 5 outer folds; (a) mean paired ΔAUC < .005 for each of the three; (b) ΔAUC(RF) > max ΔAUC of the three boosters; (c) max−min of mean AUC across tuned models ≤ same at defaults. SUPPORTED only if all three; report each |
| H17 | Soft-voting and stacking ensembles are each equivalent to the best single model on D3 and neither is superior by more than .005 | 4 of 5 boosters are already equivalent | Family = {voting vs best, stacking vs best}, Holm over 2 (members reported separately, Holm over 10). Equivalent if TOST Holm p < .05; "superior by > .005" if the one-sided test of Δ > +.005 rejects at Holm α=.05. SUPPORTED if both equivalent and neither superior |
| H18 | EBM is equivalent to the best boosting model; the WoE scorecard LR is not equivalent (≥ .005 below). Both with and without interest_rate | Additive shape functions capture most signal; WoE bins lose within-bin information | Primary = with interest_rate; 5 repeats; vs best of {GB, XGBoost, LightGBM, CatBoost}. EBM: TOST equivalent. Scorecard: SUPPORTED as "≥.005 below" if mean Δ ≤ −.005 and TOST fails; INCONCLUSIVE if TOST fails and the Δ CI includes 0 |
| H19 | On the 667k mature population: (a) calibrated \|t*−π\| < .02, (b) \|tuned recall none − balanced\| < .01, (c) mean AUC of LR, LightGBM, XGBoost differs from the D3-200k values by < .01 | Properties of the score scale, not of sample size | (a) every (model, calibrator) mean; (b) mean over models; (c) each of the three. SUPPORTED only if all three |

**Quick checks (task A) are descriptive, not tested hypotheses.** Fixed design choices: SMOTENC (categorical = nominal columns, ordinal codes) for tree models and plain SMOTE after one-hot for LR; sampling_strategy=1.0 for both methods; tuning objective = inner 3-fold log loss on a seeded 100k subsample, 25 trials per (model, outer fold), n_estimators in the search space (no early stopping); EBM fit on a seeded 100k subsample if a 128k-row fit exceeds ~2 min; WoE: 10 quantile bins, additive smoothing .5.

## Phase 3C results (2026-10-16; outcomes reported against the frozen Phase 3C pre-registration)
| id | verdict | key numbers |
|---|---|---|
| H9 equivalent boosting majority (AUC ±.005, N–B, Holm) | **SUPPORTED** | D3: 4/5 boosting models equivalent to best (LightGBM .6832): XGBoost −.0003, GB −.0005, CatBoost −.0011 (TOST Holm p<.001); AdaBoost −.0062 [−.0076, −.0049] is outside the band, as are LR −.0057, NN −.0032 (CI crosses margin), RF/ET/DT −.026…−.030 |
| H10 \|t*−π\|<.02 in all ablation cells | **SUPPORTED** | max over (cell, model, calibrator) of mean \|t*−π\| = .0121 (cell E); mean .0034–.0049 per cell |
| H11 removing rate+score: ΔAUC>.005 and larger capacity gain | **SUPPORTED** | ΔAUC(A−F) = .0483 (.6773 → .6289); default-risk dollar-capacity gain (k=50%, P2-3%) −$39.1k (A) → +$49.3k (F) |
| H12 dollar-capacity gain < count-capacity gain (expected profit) | **SUPPORTED** | $191.6k vs $102.5k per 1000 (mean over k = 30–90%); difference −$89.0k, 95% CI [−90.4k, −87.7k] (−46%) |
| H13 default-risk < random under ROI, and rate-only ≈ best model | **REJECTED** (a holds, b fails) | (a) 10/10 models negative ROI gain (−.0041…−.0077); (b) ROI(rate, high first) − ROI(best = expected-ROI XGBoost) = −.0092, CI [−.0119, −.0062], margin ±.0012 |
| H14 cheap recalibration fix ≥30% ECE cut in ≥2 of 3 folds | **REJECTED** | isotonic on trailing 6 months: +37.6% / −3.8% / +23.5% (1 of 3 folds); Platt +1.6% / +1.4% / 0.0%; EM prior shift −1237% / −305% / −511% (diverges); oracle π_T (not eligible): +26.6% / +69.3% / +46.4% |

**A. Stats** (`phase3c_stats.csv`). Best by mean AUC: D3 LightGBM (.6832), D2 LightGBM (.7081), D1 CatBoost (.7026). Equivalent set (Holm TOST<.05): D3 {XGBoost, GB, CatBoost}; D2 {XGBoost, GB, CatBoost, NN}; D1 {XGBoost, LightGBM, GB, LR} (NN and AdaBoost not). Paired-bootstrap check agrees (e.g. D3 LR [−.0058, −.0006]). OOT folds (seed-42 scores, seed sd of ΔAUC ≤.0006): CatBoost best in all three; LightGBM −.0083/−.0041/−.0035 and XGBoost −.0061/−.0041/−.0031 (inside ±.005 only in 2014–15); RF −.026…−.039. Brier and tuned-F1 differences vs best are significant (Holm) for LR, DT, RF, AdaBoost, ET on D3.
- *Earlier claims:* "no clear winner" **CONFIRMED** (D3/D2/D1 equivalent sets of 3–4 models). "CatBoost is best" **REJECTED** as a robust claim: nominal best only on D1 and OOT, third on D3/D2 and equivalent to LightGBM/XGBoost/GB in-time; OOT lead over LightGBM/XGBoost is ≤.004 AUC except 2013.

**B. Dollar capacity and ROI** (D3, P2-3%, dollar capacity, k=50%; approve-all profit $763k per 1000, ROI .0582): default-risk ROI .0519 (gain −.0063; profit gain −$41k, dollar-weighted bad rate .083); expected profit ROI .0769 (+.0187; +$123k; bad .159); expected ROI .0815 (+.0233; +$153k; bad .170); interest rate high-first ROI .0714 (+.0133; +$87k; bad .214); interest rate low-first ROI .0448 (−.0134; −$88k; bad .085). Count-capacity expected-profit ranking approves 68.7% of dollars at k=50% of loans (loan-size confound). Model-rank Spearman across k: .63 (default-risk) / .80 (expected profit) / .95 (expected ROI); across OOT folds .13 / .16 / .42. OOT ROI at k=50% (mean of 10 models; approve-all .099/.055/.031): default-risk .085/.062/.039, expected profit .125/.063/.034, rate high-first .123/.055/.026.

**C. Ablation** (`phase3c_ablation_cells.csv`, D3, 5 models × 5 repeats):
| cell | ROC-AUC | PR-AUC | Brier | ECE | mean \|t*−π\| calib. | default-risk dollar gain (k=50%, P2-3%) |
|---|---|---|---|---|---|---|
| A rate in, fico (headline) | .6773 | .2559 | .1197 | .0099 | .0048 | −$39.1k |
| B rate in, no score | .6757 | .2545 | .1199 | .0102 | .0049 | −$42.8k |
| C rate in, grade-derived | .6771 | .2563 | .1198 | .0111 | .0046 | −$38.7k |
| D rate out, fico | .6506 | .2326 | .1216 | .0096 | .0042 | **+$30.4k** |
| E rate out, grade-derived (partial removal) | .6770 | .2560 | .1198 | .0100 | .0046 | −$38.3k |
| F rate out, no score | .6289 | .2190 | .1229 | .0115 | .0034 | **+$49.3k** |
SHAP (LightGBM, rep 0, 2,000 test rows): cell A interest_rate 36.4% of mean|SHAP|, then annual_income 11.7%, loan_amount 8.1%, dti 7.5%, fico_mean 6.3%; cell F annual_income 17.5%, credit_inquiries 14.7%, loan_amount 13.2%, dti 11.7%. The grade-derived score reconstructs the rate (E ≈ A), so only D/F are genuine removals.

**D. Drift** (mean of 5 models × 3 seeds, ECE on year T): raw .041/.022/.034; isotonic-T−1 .022/.036/.022; Platt .022/.036/.022; trailing-6-month isotonic .014/.038/.017; EM prior shift .299/.146/.133; oracle .016/.011/.012. EM estimates of π_T wander (e.g. LightGBM: .40/.03/.003 vs true .156/.184/.149): verified on synthetic pure prior shift (.2509 vs true .25), so the failure is real. *(Wording corrected in Phase 3D: the earlier "mostly covariate shift" explanation was not established; see the Phase 3D EM diagnostic: EM recovers π within ≤.013 when calibration and EM batch come from the same period and fails only across years, which implicates a change in P(y | score) — calibration drift — that the prior-shift model cannot represent; covariate shift and concept drift are not separable with these data.)* Oracle π_T requires information a deployed system would not have; trailing-6-month isotonic needs labels from the recent window (available, with hindsight).

## Phase 3D results (2026-10-16; outcomes reported against the frozen Phase 3D pre-registration)
| id | verdict | key numbers |
|---|---|---|
| H15 resampling | **NOT SUPPORTED** — (i) holds, (ii) holds, (iii) fails | (i) undersampling moves raw Youden t* to ≈.49–.51 for all 4 models; SMOTE moves it toward .5 in 4/4 but weakly for RF/LightGBM/XGBoost (.158→.204, .155→.173, .147→.178; LR .144→.493). (ii) max mean \|t*−π\| calibrated: undersample .0072, SMOTE .0076. (iii) AUC vs unweighted (TOST ±.005, Holm over 8): equivalent — LR-undersample −.0002, LR-SMOTE −.0030, LightGBM-undersample −.0022, XGBoost-undersample −.0018; not equivalent — RF-undersample **+.0065** [.0034, .0096], RF-SMOTE −.0089, LightGBM-SMOTE −.0174, XGBoost-SMOTE −.0175 (CIs exclude 0) |
| H16 tuning | **SUPPORTED** | (a) ΔAUC tuned−default: LightGBM +.0012 [.0007, .0017], XGBoost +.0005 [−.0004, .0015], CatBoost +.0023 [.0011, .0035], all <.005; (b) RF +.0224 [.0208, .0240] (largest); (c) AUC spread across 5 models .0068 tuned vs .0254 default |
| H17 ensembles | **SUPPORTED** | voting +.0002 [−.0008, .0011], stacking +.0007 [−.0001, .0015] vs LightGBM (.6832): both equivalent (Holm TOST p<.001), neither superior by >.005 (Holm p=1.0) |
| H18 interpretable | **SUPPORTED** | with rate (5 repeats; best boosting LightGBM .6841): EBM −.0009 [−.0029, .0011] equivalent (Holm p=.005); WoE scorecard −.0085 [−.0105, −.0064] not equivalent, ≥.005 below. Without rate (best LightGBM .6582): EBM −.0002 equivalent; scorecard −.0167 not equivalent |
| H19 scale | **SUPPORTED** | 667k: max mean \|t*−π\| calibrated .0056; \|tuned recall none−bal\| mean .0095 (LR .017, LightGBM .010, XGBoost .005, CatBoost .007); AUC vs 200k: LR +.0043, LightGBM +.0064, XGBoost +.0067 (CatBoost +.0047), all <.01 |

**A. Quick checks (descriptive).**
- *OOT capacity gain over random, P2-3%, dollar capacity* (mean over 5 models × 3 seeds; 400-resample paired bootstrap): 2013 default-risk **−$108k/−$106k/−$70k** at k=30/50/70% (CIs exclude 0 in 15/15 model-seed cells, below zero); expected profit +$183k/+$193k/+$137k, expected ROI +$172k/+$187k/+$150k (15/15 above 0). 2014: default-risk +$13k (10/15 cells CI>0)/+$53k/+$75k (15/15); expected profit +$64k/+$63k/+$47k (15/15). **2015 (36m): default-risk +$30k/+$54k/+$66k (15/15 CIs exclude 0, above zero); expected profit +$21k/+$23k/+$20k (15/15); expected ROI +$22k (12/15)/+$29k/+$29k (15/15).** So in 2014–2015 default-risk ranking *beats* random and is at or above the expected-profit ranking at k=50–70%; only 2013 (and D3 in-time) show the negative-gain "paradox". The D3-in-time and 2013 results of 3B/3C therefore do not generalise to later vintages; the cause is not established (2013 includes 60-month loans; D3 mixes vintages) and is an open question.
- *EM diagnostic (5 models × 3 seeds):* (a) calibrate on half of T−1, EM on the other half: mean |π̂ − π| .003–.013 (π̂ .1632 vs .1614 in 2013, .1574 vs .1567, .1341 vs .1371); (b) random halves of year T: .005–.012; (c) calibrate on T−1, EM on year T: π̂ .455/.035/.015 vs true .156/.185/.149 (|err| .30/.15/.13; mean calibrated p .178/.148/.127 vs π_T). None of (a)/(b) hit the 1000-iteration cap (135–175 iterations); (c) hit it in 20% of 2014 cells. **Reading:** EM is fine when calibration and the EM batch come from the same period, so the failure is not simply a weak classifier; it appears only across years, where mean(p) is already off by .02–.04 from π_T, i.e. P(y | score) changed. This is calibration drift; our data cannot separate covariate shift from concept drift, so the 3C wording "mostly covariate shift" is withdrawn (see correction in the 3C section).

**B. Ensembles** (`phase3d_ensembles*.csv`, D3, 10 repeats, no retraining; split indices asserted identical across models): AUC / Brier / ECE / mean|t*−π|: stacking .6839 / .1190 / .0054 / .0023; voting .6833 / .1191 / .0066 / .0034; LightGBM .6832 / .1191 / .0058 / .0048; XGBoost .6829; CatBoost .6820; LR .6775; RF .6571. Each ensemble is also equivalent to LightGBM, XGBoost and CatBoost individually and not equivalent to LR (+.006) or RF (+.026). **Title:** "Ensemble" cannot be justified as a *combination* contribution (best ensemble gain +.0007 AUC, −.0001 Brier). It may stay only if it refers to tree-ensemble learners (the GBDT/RF family), not to voting/stacking.

**C. Interpretable baselines** (`phase3d_interp*.csv`, 5 repeats; EBM full 128k-row folds, ~6.5–8 min per unit of 6 fits; no subsample needed): with rate — EBM AUC .6832, Brier .1191, ECE .0054, |t*−π| .0028; WoE scorecard .6757, .1199, .0061, .0041. Without rate — EBM .6580, WoE .6415 (best boosting .6582). Scorecard top-5 features by |coefficient| (rep 0): num_credit_lines, loan_amount, interest_rate, total_credit_lines, employment_length (`phase3d_scorecard_points.csv`).

**D. Resampling** (`phase3d_resampling*.csv`; D3 × 5, D1 × 3; resampling inside training folds only; SMOTENC with ordinal nominal codes for trees, plain SMOTE after one-hot for LR; tree models in the resampled variants use ordinal codes, not LightGBM category dtype). *Does threshold moving match or beat resampling?* Yes: on D3 the unweighted model with a Youden threshold gets F1/MCC .331/.183 (LR), .336/.190 (LightGBM), .317/.162 (RF), .334/.188 (XGBoost); random undersampling at 0.5 gives .331/.183, .333/.187, .321/.167, .333/.186 (equal; RF +.004 F1), and SMOTE at 0.5 gives F1 .007–.043 for the three tree models (SMOTE barely moves their probabilities), .322/.171, .322/.169 with Youden for LightGBM/XGBoost (≈.014 F1 below unweighted+Youden). D1 shows the same pattern.

**E. Scale** (`phase3d_scale*.csv`; wall-clock: D3 667k scoring 248 s for 4 models × 2 weightings × 3 repeats, strict 1.345M scoring 49 s, analysis 105 s). 667k π=.1483; AUC: LightGBM .6896, XGBoost .6896, CatBoost .6867, LR .6818. Sensitivity (1.345M strict, π=.1996; 2016+ censored, 1 repeat, threshold/calibration only): calibrated |t*−π| ≤.012 (LR Platt .0124, LightGBM isotonic .0062, balanced .0004–.0032); balanced raw t* ≈.5.

**F. Optuna nested CV** (`phase3d_optuna.csv`, `optuna_params.json`; 25 trials × 5 outer folds × 5 models, inner 3-fold on a seeded 100k subsample, log-loss objective; 5 parallel processes with 2 threads each; wall-clock ≈ 1 h 15 min, summed tuning time 2.1 h):
| model | AUC default | AUC tuned | ΔAUC [95% t-CI over 5 folds] | ΔBrier | \|t*−π\| default → tuned |
|---|---|---|---|---|---|
| LR | .6786 | .6786 | −.0000 [−.0001, .0000] | −.0000 | .0086 → .0080 |
| LightGBM | .6842 | .6854 | +.0012 [.0007, .0017] | −.0001 | .0061 → .0038 |
| XGBoost | .6843 | .6848 | +.0005 [−.0004, .0015] | −.0001 | .0054 → .0054 |
| Random Forest | .6589 | .6813 | +.0224 [.0208, .0240] | −.0026 | .0055 → .0080 |
| CatBoost | .6832 | .6854 | +.0023 [.0011, .0035] | −.0002 | .0050 → .0059 |
Best parameters are stable for CatBoost (n_estimators 502 cv .02, depth 7 cv .10) and LightGBM (lr .0215, n_estimators 407) but not for regularisation terms (reg_lambda cv >1.3; min_child_samples cv .68). Tuned RF picks min_samples_leaf ≈33 (cv .6) and max_features ≈.38 — the leaf-purity effect again. Tuned RF (.6813) remains .004 below the tuned boosters.

## Phase 4 PRE-REGISTRATION (written 2026-10-17, committed BEFORE tasks A–C; H5–H19 remain frozen)
_Wording rule: "not equivalent" is not "worse". If a TOST fails but the difference CI includes 0 the cell is labelled INCONCLUSIVE._

| id | Hypothesis | Decision rule (fixed now) |
|---|---|---|
| H20 | Restricted to 36-month loans only, the 2013 default-risk-ranking dollar-capacity gain over random (P2-3%, k=50%) is still below the 2014 and 2015 gains, i.e. term mix alone does not explain the sign change | Cells = mean over 5 models × 3 seeds of the OOT score cache, 36m test loans only. SUPPORTED if gain(2014) − gain(2013) and gain(2015) − gain(2013) are both > 0 with 95% bootstrap CIs excluding 0 (independent bootstrap across vintages). INCONCLUSIVE if both point differences are > 0 but a CI includes 0. REJECTED if either point difference is ≤ 0 (term mix explains the sign change). Direction reported either way |
| H21 | Mean-\|SHAP\| feature-rank Spearman is ≥ .8 for every pair among {LightGBM, XGBoost, CatBoost, GB}, ≥ .6 between each of those four and each of {LR, WoE}, and interest_rate ranks first in every model that includes it | D3, unweighted, raw scores, 3 repeats × 2,000 seeded test rows; with-rate cell is primary. Ranks over original features (one-hot dummies summed within feature before ranking). Spearman computed per repeat then averaged; the threshold applies to the mean. SUPPORTED only if all three clauses hold; each reported separately. Scales differ (log-odds vs probability for RF) — ranks only |
| H22 | For ≥ 70% of test loans, the top-3 risk-increasing reason codes overlap in ≥ 2 of 3 features between tuned LightGBM (TreeSHAP) and the EBM | 2,000 seeded test rows × 3 repeats, pooled. Primary = interest_rate allowed; the version with interest_rate excluded from reason codes (model still uses it) is reported separately with its own verdict. Risk-increasing = positive contribution to log-odds of default; loans with < 3 positive contributions use the available ones and overlap is capped at their count (reported). SUPPORTED if the pooled share ≥ .70 |

**Fixed design choices.** Term/vintage split uses the existing OOT score cache (`.cache/oot_scores`), rankings default-risk, expected-profit and interest-rate-only; k ∈ {30,50,70}%; paired bootstrap 1,000 draws. SHAP: TreeExplainer (trees; RF tuned with `optuna_params.json` medians), LinearExplainer (LR), native additive term contributions (EBM, interactions=0 as in 3D), exact points (WoE, coef × WoE). Reason-code labels live in `creditrisk/reason_codes.py`. No regulatory-compliance claim is made.

## Phase 4: Explainability  (done 2026-10-17)
- [x] Term and vintage diagnostic (task A) — `DONE` — `experiments/term_vintage_check.py`, `phase4_term_vintage.csv`, `phase4_h20.csv`, figure `phase4_term_vintage.png`; **H20 SUPPORTED**
- [x] SHAP on all models with 2000 rows × 3 repeats — `DONE` — `experiments/shap_study.py`, `phase4_shap*.csv`, figures `phase4_shap_top10.png`, `phase4_shap_rankcorr.png`
- [x] Rank correlation of attributions across models — `DONE` — **H21 SUPPORTED**
- [x] Top-k reason codes (EBM vs tuned LightGBM) — `DONE` — `creditrisk/reason_codes.py`, `experiments/reason_codes_study.py`, `phase4_reasoncodes*.csv`, `phase4_reasoncode_examples.{csv,md}`; **H22 SUPPORTED**

### Phase 4 results (2026-10-17; outcomes against the frozen Phase 4 pre-registration, commit b3240f4)
| id | verdict | key numbers |
|---|---|---|
| H20 36m-only 2013 default-risk capacity gain < 2014 and 2015 | **SUPPORTED** | 36m, P2-3%, k=50% (USD per 1000 loans, 5 models × 3 seeds): 2013 −$33.3k, 2014 +$2.9k, 2015 +$54.3k. gain(2014)−gain(2013) = +$36.1k [26.0k, 46.7k]; gain(2015)−gain(2013) = +$87.6k [77.8k, 97.1k]. Term mix alone does not explain the sign change |
| H21 attribution agreement | **SUPPORTED** (all 3 clauses) | booster pairs (LightGBM, XGBoost, CatBoost, GB): min Spearman .961 (≥.8); each booster vs LR/WoE: min .845 (≥.6; WoE is the lowest, .83–.89; LR .93–.95); interest_rate ranks first in all 8 models in every repeat |
| H22 top-3 reason-code overlap ≥2/3 for ≥70% of loans | **SUPPORTED** (both versions) | pooled share: rate allowed .989 (per-rep .973–.984), rate excluded .987; mean Jaccard .83/.82; top-1 code identical for ~.86 (rate allowed) / ~.82 (excluded) |

**A. Term × vintage** (`phase4_term_vintage.csv`; default-risk gain, P2-3%, k=50%, mean of 5 models × 3 seeds, Poisson-bootstrap CI on the averaged per-loan column; all-term rows reproduce the 3D quick-check numbers exactly):
| test vintage | term | n test | gain, default-risk | gain, expected profit | gain, rate-only | mean rate % | default rate | corr(p, rate) |
|---|---|---|---|---|---|---|---|---|
| 2013 | all | 134,804 | −$105.9k [−117.3, −93.3] | +$193.4k | −$179.3k | 14.5 | .156 | .82 |
| 2013 | 36 | 100,422 | −$33.3k [−41.4, −25.3] | +$66.6k | −$74.7k | 13.4 | .123 | .81 |
| 2013 | 60 | 34,382 | +$34.5k [−2.1, +67.9] (n.s.) | +$207.3k | −$77.3k | 17.9 | .251 | .77 |
| 2014 | all | 223,103 | +$53.4k [43.9, 61.8] | +$63.0k | +$2.9k (n.s.) | 13.7 | .184 | .84 |
| 2014 | 36 | 162,570 | +$2.9k [−3.6, +9.8] (n.s.) | +$44.3k | −$27.0k | 12.5 | .137 | .82 |
| 2014 | 60 | 60,533 | +$227.9k [200.7, 254.8] | +$138.8k | +$112.3k | 16.9 | .311 | .81 |
| 2015 | 36 | 283,026 | +$54.3k [49.5, 59.6] | +$22.6k | +$32.1k | 11.3 | .149 | .79 |
At k=30% and 70% the signs agree except 2014-36m (k=30%: −$16.9k; k=70%: +$17.1k) and 2013-60m (k=30%: +$1.3k n.s.; k=70%: +$45.1k, CI excludes 0).
What the cells suggest (**hypotheses, not tested here**): (i) the sign change is a vintage effect within 36-month loans, so the 2013 → 2015 pattern is not a 60-month-share artefact; (ii) for 36m loans the default rate rises .123 → .137 → .149 while the mean rate falls 13.4 → 12.5 → 11.3%, i.e. the price-for-risk margin compresses, so selecting on low risk stops giving up as much interest income as it saves in losses; (iii) 60-month loans carry the largest default-risk gain in 2014 (+$228k), consistent with high-default, long-duration loans being where the ranking pays; (iv) corr(p, rate) is .77–.84 in every cell, so the score is largely the rate everywhere and this correlation does not by itself explain the change. Not tested: 2013/2014 models were fit on mixed terms and the 2015 model on 36m only, so vintage is confounded with training composition; a 36m-only refit of the 2013 and 2014 folds would remove that confound.

**B. SHAP/attribution agreement** (D3, unweighted, raw scores, 3 repeats × 2,000 seeded test rows; TreeSHAP for LightGBM/XGBoost/GB/tuned RF, CatBoost native ShapValues, shap LinearExplainer for LR, native additive term scores for EBM (interaction terms split equally between their two features), exact centred points coef × WoE for the scorecard; dummies summed to the original feature; RF is on the probability scale, all others log-odds, so ranks/shares only). Mean share of mean |attribution| (%):
| feature | LightGBM | XGBoost | CatBoost | GB | RF tuned | LR | WoE | EBM |
|---|---|---|---|---|---|---|---|---|
| interest_rate | 36.5 | 36.5 | 37.3 | 37.1 | 35.3 | 29.0 | 38.3 | 32.9 |
| annual_income | 11.5 | 11.1 | 10.9 | 11.5 | 11.0 | 7.6 | 7.3 | 11.7 |
| loan_amount | 8.1 | 8.3 | 6.5 | 8.3 | 7.3 | 8.0 | 5.3 | 9.1 |
| debt_to_income | 7.4 | 7.3 | 9.1 | 7.6 | 9.6 | 7.2 | 8.9 | 6.9 |
| fico_mean | 6.4 | 6.2 | 8.2 | 6.4 | 9.5 | 10.0 | 6.6 | 7.0 |
Ranks 2–4 differ by model (annual_income #2 in 6 of 8; fico_mean #2 for LR; debt_to_income #2 for the WoE scorecard). Pairwise Spearman (Kendall) range .83–.99 (.63–.96); lowest pairs involve the WoE scorecard (vs LR .83, vs GB .85, vs EBM .86). Stability across the 3 repeats: mean Spearman of ranks .97–1.00; top-3 set identical in all repeats for LightGBM, XGBoost, CatBoost, WoE, EBM (not GB, RF, LR). With vs without rate (shared features): LightGBM Spearman .84 (cell D) / .83 (cell F), EBM .87 / .84. Without rate: cell D (score kept) fico_mean becomes first (LightGBM 21.8%, EBM 19.6%); cell F (rate and score out) annual_income 17.2%, credit_inquiries 14.9–14.2%, loan_amount 13.1–14.2%, dti 10–12%.
*Conference-paper SHAP claims (wording from the author's prompt; the paper's own SHAP output was not re-derived):* "interest rate is the top feature" **CONFIRMED** (first in all 8 models, 29–38%, every repeat); "credit score and DTI complete the top 3" **WEAKENED**: fico_mean is top-3 only for LR (#2) and fico/DTI are #4–#6 in most models (annual_income and loan_amount fill ranks 2–3 in the boosters; DTI is top-3 for CatBoost, tuned RF and the scorecard); "attributions are similar across models" **CONFIRMED** for ranks (Spearman ≥.83 for every pair, ≥.96 among boosters), with the caveat that this is largely one dominant feature (interest_rate, ~35%) and the remaining 15 features share a flat 1–12% tail; the paper's 50-sample global SHAP for one model could not support a cross-model claim.

**C. Reason codes** (`phase4_reasoncodes.csv`, `_jaccard.csv`, `_examples.csv/.md`; 3 reps × 2,000 loans pooled; "risk-increasing" = positive contribution to log-odds of default; primary EBM = 3D configuration with 10 interactions; the no-interaction EBM is a sensitivity: .989/.986 for k=3).
| version | k | share with ≥2 shared codes (of the codes both models issue) | strict (≥2 shared) | mean Jaccard | loans with <k positive contributions |
|---|---|---|---|---|---|
| rate allowed | 3 | .989 | .978 | .83 | 4.6% |
| rate allowed | 4 | .986 | .986 | .84 | 11.3% |
| rate excluded | 3 | .987 | .974 | .82 | 5.3% |
| rate excluded | 4 | .983 | .983 | .84 | 13.6% |
Jaccard (k=3, rate allowed): 5th–25th percentile .5, median 1.0. interest_rate is a top-3 code for ~51% of loans in both models and the top code for ~41–42%. Chance level for a 2-of-3 match drawing 3 of 16 features at random ≈ 7%. Disagreement concentrates in low-risk loans whose positive contributions are tiny (example 5: all < .03 log-odds). Two worked examples (from `phase4_reasoncode_examples.md`, amounts rounded): (1) loan 8,500, income 20,000, DTI 19.2, rate 15.6%, FICO 662, employment length missing → both models: interest rate (+.435 EBM / +.397 SHAP), low income (+.314/+.305), employment length not reported (+.311/+.276), overlap 3/3; (2) loan 19,000, income 65,000, rate 15.9%, FICO 667, purpose house → both: interest rate (+.472/+.490), loan amount (+.208/+.208); third code differs (EBM loan purpose +.119, LightGBM FICO +.113), overlap 2/3.
**Interpretation limits:** labels in `creditrisk/reason_codes.py` assume that a positive contribution means "high" (or "low" for income/score/employment) on that feature; for non-monotone features (loan_amount, num_credit_lines, loan_purpose, home_ownership) the label states the feature, not a direction, and example 5 shows a code ("loan amount is high" for a 12,000 loan, +.013) that a direction-blind template would misdescribe. interest_rate is the lender's own price: excluded-version codes are listed separately because "your interest rate" is not an actionable borrower attribute.

## Phase 5 PRE-REGISTRATION (written 2026-10-17, committed BEFORE tasks B–F; H5–H22 remain frozen)
_Wording rule: "not equivalent" is not "worse"; CI includes 0 but TOST fails = INCONCLUSIVE._

| id | Hypothesis | Decision rule (fixed now) |
|---|---|---|
| H23 | Refitting the 2013 and 2014 OOT folds on 36-month loans only (train, calibration and test restricted to 36m, as 2015 already is) leaves the sign pattern of the default-risk dollar-capacity gain (P2-3%, k=50%) unchanged: 2013 below 2014 and 2015 | LR, RF, LightGBM, XGBoost, CatBoost × 3 seeds; gain = mean over models × seeds; 1,000-draw Poisson bootstrap. SUPPORTED if gain(2014) − gain(2013) > 0 and gain(2015) − gain(2013) > 0 with both 95% CIs excluding 0 (independent bootstrap across vintages). INCONCLUSIVE if both point differences > 0 but a CI includes 0. REJECTED if either point difference ≤ 0. No cause for the sign change is claimed in the paper unless H23 is SUPPORTED and a further test isolates the cause |
| H24 | Serving parity: predictions from the saved, reloaded deployment pipeline match the in-memory pipeline to within 1e-9 on 1,000 held-out loans, and the app's /api/predict returns the same probability as the library call for all of them | max abs difference < 1e-9 for (a) in-memory vs reloaded (calibrated PD and EBM score); (b) library `predict` vs JSON `/api/predict` value compared at the response precision (JSON carries the float repr, so tolerance 1e-9 holds). SUPPORTED only if both hold for all 1,000 loans |

**Fixed design choices.** Task A: same OOT protocol and score code as `oot_scores.py` (fit ≤ T−2, calibrate/threshold on T−1, test T), all three sets restricted to term = 36 for every T; 3 seeds; rankings default-risk (isotonic-calibrated), expected-profit and rate-only; k ∈ {30,50,70}%. Deployment model (task B): tuned LightGBM with numeric parameters = median over the 5 outer folds in `optuna_params.json` (integers rounded; categorical = mode), unweighted, 5-fold OOF isotonic calibration, Platt behind a config switch; EBM companion = 3D configuration (interactions=10). Tiers (task C): PD < 0.75π, 0.75π–1.5π, > 1.5π, review tiers not accept/reject cutoffs. Prototype status is stated in the app and model card.

### Phase 5A results: 36-month refit (2026-10-17; `experiments/term36_refit.py`, `phase5_term36.csv`, `phase5_h23.csv`)
Training, calibration and test all restricted to 36-month loans; 5 models × 3 seeds; P2-3%, dollar capacity; gain in USD per 1000 loans, Poisson-bootstrap 95% CI on the model-and-seed average. Training sizes: 29,096 (T=2013; `mortgage_accounts` dropped as >50% missing), 72,566 (2014), 172,988 (2015).
| T | default-risk k=30 / 50 / 70% | expected-profit k=50% | rate-only k=50% | default rate | mean rate % |
|---|---|---|---|---|---|
| 2013 | −$36.0k / **−$21.2k** [−28.6, −12.8] / −$5.9k [−14.3, +2.3] | +$62.4k | −$74.7k | .123 | 13.4 |
| 2014 | −$7.5k / **+$11.6k** [+4.8, +18.0] / +$22.7k | +$43.3k | −$27.0k | .137 | 12.5 |
| 2015 | +$30.0k / **+$54.3k** [49.2, 59.4] / +$66.0k | +$22.6k | +$32.1k | .149 | 11.3 |
**H23: SUPPORTED.** gain(2014) − gain(2013) = +$32.8k [21.8k, 43.2k]; gain(2015) − gain(2013) = +$75.5k [65.6k, 84.8k]. The sign pattern survives the 36-month refit (mixed-term fit: 2013 −$33.3k, 2014 +$2.9k). The 2014 gain is larger with the 36m-only fit (+$11.6k, CI now excludes 0). The 2013 fold now trains on only 29k loans (2010–2011 36m), so its level is noisy. 2015 is identical to the earlier run by construction. **No cause is claimed:** the pre-registered rule requires a further test that isolates one, and none has been run (the price-for-risk margin compression remains a hypothesis).

## Phase 5: Deployment  (done 2026-10-17; code on branch `app-v2`, study results on `journal-extension`)
- [x] 36-month refit (H23) — `DONE` (journal-extension) — **H23 SUPPORTED**
- [x] Persist the pipeline with joblib (versioned, hash/schema/version-validated load) — `DONE` — `creditrisk/serving.py`, `scripts/train_deploy.py`, `models/deploy/`
- [x] Calibrated PD output (isotonic default, Platt switch) — `DONE`
- [x] Review tiers anchored on π; old hardcoded 15/30/50/70% bands and the 0.5 cutoff removed — `DONE` — `creditrisk/tiers.py`
- [x] Rewrite `app.py` and templates (no training at request time, strict validation, direction-aware reason codes, prototype notice) — `DONE`
- [x] Tests (`tests/`) — `DONE` — 39 passed
- [x] Dockerfile / docker-compose / .dockerignore / README — `WRITTEN, NOT BUILT` (Docker is not installed on this machine; no claim that the image builds or runs)

### Phase 5 results (outcomes against the frozen Phase 5 pre-registration, commit f60b319)
| id | verdict | key numbers |
|---|---|---|
| H23 36-month refit keeps 2013 < 2014, 2015 | **SUPPORTED** | see Phase 5A results above: −$21.2k / +$11.6k / +$54.3k at k=50%; differences +$32.8k [21.8, 43.2] and +$75.5k [65.6, 84.8]. No cause claimed |
| H24 serving parity | **SUPPORTED** | 1,000 held-out loans: reloaded vs in-memory max |Δ| = 1.1e-16 (calibrated PD) and 1.1e-16 (EBM probability); `/api/predict` vs library call max |Δ| = 0.0 (PD and EBM probability); tiers and reason-code features identical for all 1,000 |

**Persistence** (`models/deploy/`, trained by `python scripts/train_deploy.py`, 495 s, from clean commit `f27d7d3` on app-v2): `bundle.joblib` 1.38 MB (LightGBM pipeline + calibrator + EBM + feature list/dtypes + π + medians + 0.1/99.9% ranges), `metadata.json` (versions: Python 3.13.5, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, lightgbm 4.6.0, interpret 0.7.8, joblib 1.5.3; schema hash 89c547fad2da2087; training issue dates; git hash; SHA-256 of the bundle), `model_card.md`, `test_metrics.csv`, `tier_report.csv`, `reliability_15bins.csv`, `parity_sample.csv` (1,000 held-out loans, feature values only), `parity_expected_in_memory.csv`. Everything is under 5 MB and is committed; regenerate with the command above (deterministic: two runs gave identical outputs). Data: mature population D3 (667,024 loans; training split 533,619, held-out stratified 20% = 133,405; π = .1483); LightGBM params = median of the 5 outer-fold Optuna winners (numeric, integers rounded): n_estimators 408, learning_rate .0221, num_leaves 12, min_child_samples 109, subsample .788, colsample_bytree .592, reg_lambda .0724; unweighted; isotonic on 5-fold OOF scores (fresh pipeline per fold). EBM = 3D configuration (interactions=10), fit in 406 s.
**Model-card headline metrics (held-out, in-time, 1,000-resample bootstrap):** ROC-AUC .6898 [.6859, .6937]; PR-AUC .2680 [.2626, .2734]; Brier .1188 [.1177, .1200] (raw .1189); ECE .0028 [.0025, .0053] (raw .0041 [.0032, .0063]); log loss .3905; EBM ROC-AUC .6892 [.6853, .6930], ECE .0035.
**Recalibration schedule:** annual (documented in README and card). *Note on the prompt's "ECE rose 10–25× out of time":* the study data give about 8–13× for isotonic-on-previous-year (.022/.036/.022 vs this model's in-time .0028) and 5–10× for raw scores; the model card then stated 8–13×. *[Superseded in Phase 6: the card now quotes the measured H25 numbers — ECE on the new year .034/.018/.039, 6–14× the 133k in-time ECE of .0028 — and records that H25 was not supported; the bundle was regenerated byte-identically at commit 2051444.]*

**Tiers** (held-out split; edges .1112 and .2224 = 0.75π and 1.5π):
| tier | share | observed default rate [95% CI] | mean PD | observed − PD [95% CI] | within CI |
|---|---|---|---|---|---|
| 1: PD < 0.75π | .393 | .0684 [.0663, .0706] | .0692 | −.0008 [−.0031, +.0013] | yes |
| 2: 0.75π ≤ PD ≤ 1.5π | .417 | .1605 [.1573, .1639] | .1620 | −.0015 [−.0044, +.0015] | yes |
| 3: PD > 1.5π | .191 | .2858 [.2796, .2916] | .2825 | +.0033 [−.0019, +.0083] | yes |
Reliability: mean PD matches the observed rate within the bootstrap CI in all three tiers (in-time); the observed default rate is 4.2× higher in tier 3 than tier 1. This says nothing about out-of-time calibration (see limitations).

**Tests** (`python -m pytest`, 39 passed): `tests/test_creditrisk.py` (encoder/pipeline leak-freeness, calibrator monotonicity, tiers cover all PDs incl. edges, direction-aware labels, borrower view hides interest_rate, input validation, loud load failures on missing dir / version / schema / corrupted hash), `tests/test_app.py` (valid/invalid/missing/unknown-category/out-of-range/non-JSON requests, form, schema, health, pages render with the notice, old bands and 0.5 cutoff absent, figure whitelist, refuses to start without a model, no training code in app.py), `tests/test_parity.py` (H24; skipped if `models/deploy` is absent).
**Not tested:** Docker build/run (not installed here); load/concurrency; any fairness or regulatory property.

## Phase 6 PRE-REGISTRATION (written 2026-10-17, committed BEFORE task C; H5–H24 remain frozen)
| id | Hypothesis | Decision rule (fixed now) |
|---|---|---|
| H25 | The deployed configuration (tuned LightGBM, isotonic calibration fitted on out-of-fold scores) run on the 2013, 2014 and 2015 (36m) OOT folds shows ECE on the new year at least 5× its in-time ECE in at least 2 of 3 folds, and ROC-AUC 0.010–0.030 below its in-time AUC | Protocol folds of the earlier OOT study (fit on years ≤ T−2, test on year T; 36-month only for T=2015; features missing in >50% of the training window dropped), 3 seeds. For each fold: the training window is split 80/20 stratified at random; the configuration is fitted on the 80% (median Optuna parameters, 5-fold OOF isotonic) and evaluated on (a) the 20% in-time hold-out and (b) all loans of year T. "ECE ratio" = seed-mean ECE(year T) / seed-mean ECE(in-time), 15 equal-frequency bins. "AUC gap" = seed-mean AUC(in-time) − AUC(year T). SUPPORTED if the ratio ≥ 5 in ≥ 2 of 3 folds AND the AUC gap lies in [.010, .030] in all 3 folds; each clause reported separately. A 36-month-only variant of all three folds is reported as secondary and does not enter the verdict |

**Fixed design choices.** Calibration is on OOF scores of the training window (as deployed), NOT on year T−1 labels. Bootstrap CIs (1,000 resamples) on both ECEs, and on their ratio. The in-time hold-out is small for early folds (ECE bias is positive at small n; this is stated in the results, not corrected).

## Phase 7 PRE-REGISTRATION (written 2026-10-17, committed BEFORE any external-data analysis; H5–H25 remain frozen)
Applies separately to every external dataset that loads. No external data had been opened or analysed when this block was committed (`external_data/` did not exist).

| id | Hypothesis | Decision rule (fixed now) |
|---|---|---|
| H26 | Base-rate theory: after isotonic calibration on out-of-fold scores, the Youden t* is within 0.02 of the training prevalence π | n ≥ 20,000: per model, mean over repeats of \|t*−π\| ≤ 0.02. SUPPORTED if ≥ 5 of 6 models satisfy this AND the mean over models ≤ 0.02. n < 20,000 (German Credit): per model, bootstrap 95% CI of t* (1,000 resamples of the OOF set, one reference repeat) contains π; report the fraction of models; SUPPORTED if the fraction > 50% ("most") |
| H27 | Weighting: balanced weighting moves the raw (uncalibrated) Youden t* toward 0.5 relative to unweighted, and changes tuned-threshold recall by < 0.02 on average across models (< 0.04 for n < 5,000) | Clause 1: mean over models and repeats of \|t*_none−0.5\| − \|t*_bal−0.5\| > 0, with the sign positive for ≥ 5 of 6 models. Clause 2: tuned recall = recall of the isotonic-calibrated, OOF-Youden-tuned rule on the test set; mean over models of \|mean-repeat recall(balanced) − mean-repeat recall(none)\| < 0.02 (0.04 if n < 5,000). SUPPORTED only if both clauses hold; clauses reported separately |
| H28 | Model family: at default hyperparameters LightGBM, XGBoost, CatBoost and GB are each equivalent to the best model on ROC-AUC | Best = highest mean-test-AUC model among the six. Each of the four boosters (excluding the best itself if it is one of them) is compared with the best by Nadeau–Bengio corrected paired TOST, margin ±.005, Holm over the comparisons of that dataset. n ≥ 20,000: SUPPORTED if all comparisons are equivalent; NOT SUPPORTED if any comparison fails TOST AND its corrected CI excludes 0; otherwise INCONCLUSIVE (TOST fails with a CI including 0 = INCONCLUSIVE, never "worse"). n < 20,000: same labels, with INCONCLUSIVE reported wherever power is insufficient and stated as such |

**Fixed design choices.** Models: LR, RF, LightGBM, XGBoost, CatBoost, GB at default hyperparameters (as in the main study). Protocol as Phase 2: stratified train/test split, 5-fold OOF raw scores on the training part, isotonic and Platt calibrators fitted on OOF scores, thresholds (Youden) on calibrated OOF scores, weighting {none, balanced} × threshold {0.5, tuned}, evaluation on the untouched test set. Repeats: 10 for n ≥ 20,000; 30 for German Credit. Splits: n ≥ 20,000 uses the Phase 2 split fractions; German Credit uses stratified 70/30 (≈ 90 test positives, ≈ 210 training positives). π = positive rate in the training part. ECE = 15 equal-frequency bins. t* CIs: bootstrap on one reference repeat (repeat 0). No profit analysis (no cash-flow data). Label coding is verified from the data before use. Datasets: German Credit and Taiwan Default (UCI) if reachable; Give Me Some Credit only if the user placed it in `external_data/`.

**Clarifications fixed before any scoring run (code and data cards committed first; the block above is unchanged).** (1) H26 primary weighting = `none` (isotonic recalibration makes weighting nearly moot; `balanced` is reported as secondary). (2) H26 bootstrap for German Credit: 1,000 resamples of the (OOF score, label) pairs of repeat 0, refitting the isotonic calibrator and the Youden threshold in every resample. The same bootstrap is reported for Taiwan as supplementary. (3) H28 uses raw test-set ROC-AUC with weighting `none`; Holm is over the booster-vs-best comparisons of that dataset (4, or 3 if the best model is itself a booster); comparisons of the other models with the best are reported but do not enter the verdict. Nadeau–Bengio ratio n_test/n_train = .25 (Taiwan) and .4286 (German). (4) H27 clause 2 uses the isotonic-calibrated, OOF-Youden-tuned rule as pre-registered; for continuity with the Lending Club tables the raw-score-tuned recall gap (the Phase 2 definition) is also reported, as supplementary. (5) Taiwan: 10 repeats, stratified 80/20 split of all rows with 5-fold OOF on the 80%; German: 30 repeats, stratified 70/30 with 5-fold OOF on the 70%. (6) The 35 exact duplicate rows in the Taiwan file are kept. (7) Give Me Some Credit was not placed in `external_data/` and is not used.

## Phase 6: Paper support  (done 2026-10-17; no manuscript prose written)
- [x] Export tables (CSV + LaTeX booktabs) and figures (PNG 300 dpi + PDF) — `DONE` — `experiments/export_paper_assets.py` → `paper_assets/` (33 table files for T1–T15 incl. sub-tables, 9 figures, `MANIFEST.md`)
- [x] Automated verification of exported numbers and of tracker statements — `DONE` — `paper_assets/verify.py` (see results)
- [x] List of paper claims that changed vs. conference version — `DONE` — `paper_assets/claims_audit.md`, table T15
- [x] Reference candidates (all UNVERIFIED) — `DONE` — `paper_assets/references_to_verify.md`
- [x] Reproducibility package — `DONE` — `scripts/reproduce_all.sh` (dry-run validated; the full run was not re-executed)
- [x] Deployment configuration out of time (H25) — `DONE` — **H25 NOT SUPPORTED**

### Phase 6 results
**H25 (frozen at e647452): NOT SUPPORTED.** Deployed configuration (tuned LightGBM with median Optuna parameters, isotonic on 5-fold OOF scores of the training window) on the protocol OOT folds, 3 seeds; in-time = random 20% hold-out of the training window; bootstrap 1,000 resamples (rows shared across seeds, each seed with its own labels).
| fold | n in-time / OOT | AUC in-time [95% CI] | AUC OOT | AUC gap [95% CI] | ECE in-time [CI] | ECE OOT [CI] | ECE ratio [CI] |
|---|---|---|---|---|---|---|---|
| 2013 | 7,958 / 134,804 | .6961 [.6878, .7052] | .6733 | +.0228 [+.0139, +.0323] | .0141 [.0134, .0197] | .0344 [.0325, .0364] | **2.4** [1.7, 2.6] |
| 2014 | 18,631 / 223,103 | .6891 [.6837, .6949] | .6827 | +.0063 [+.0007, +.0127] | .0074 [.0083, .0123]\* | .0179 [.0160, .0192] | **2.4** [1.4, 2.2]\* |
| 2015 (36m) | 34,598 / 283,026 | .6710 [.6661, .6756] | .6826 | −.0116 [−.0173, −.0062] | .0067 [.0067, .0099] | .0387 [.0374, .0399] | **5.8** [3.9, 5.8] |
\*The bootstrap in-time ECE is biased upward at small n, so the point estimate lies at or below the CI's lower edge in 2014 and 2015; ratios are the seed-mean point estimates.
Clause 1 (ECE ratio ≥ 5 in ≥ 2 of 3 folds): **fails**, 1 of 3 (2015 only). Clause 2 (AUC gap within [.010, .030] in all folds): **fails** (.023 in range; .006 below; −.012 negative, the 2015 OOT year is *easier* than the in-window hold-out). 36-month-only variants (secondary, not in the verdict): AUC gap +.029/+.003, ECE ratio 1.4/0.6.
Reading: the configuration's calibration is clearly worse on the new year (ECE .018–.039; 6–14× the .0028 of the 133k in-time split), but against its *own small in-window hold-outs* the ratio is only 1.4–5.8×, because small hold-outs inflate ECE; and AUC does not consistently fall out of time for this configuration. The model-card sentence was rewritten with these measured numbers (app-v2).

**Export and verification.** 42 assets exported (33 tables for T1–T15 with sub-tables, 9 figures); `paper_assets/verify.py`: **68 checks passed, 0 table mismatches**, 1 tracker discrepancy (the H4 "J lost ≤ .007" statement; corrected above). Checks that first failed and were fixed in the export/verify code, not in the data: H10/H19 scorecard numbers initially included raw (uncalibrated) rows; a bootstrap-alignment bug in the first H25 run (in-time CIs; the point estimates and verdict were unaffected). Clarifications found by verification: H12's $191.6k/$102.5k are means over k = 30–90% (k=50%: $242.1k/$122.9k); the 3A AUC-drop range .016–.021 is .015–.021 on the 3-seed CSV.
**Repository hygiene** (no secrets; see the decisions log for the list): largest tracked file 1.6 MB, `.git` 6.8 MB; no tracked dataset, pickles or caches; a Windows co-author's account name appears in the ignored `.venv/pyvenv.cfg`; untracked `venv.txt` contains a local absolute path; the legacy `app.py` runs with `debug=True` (127.0.0.1 only), v2 does not.

### Open items before manuscript drafting
1. Fill in the Lending Club data source (URL/licence/date accessed) in `scripts/reproduce_all.sh` and the paper's data statement.
2. Verify every reference in `paper_assets/references_to_verify.md` yourself (all UNVERIFIED).
3. Locate the manuscript numbers behind each row of `claims_audit.md` (the manuscript is not in the repository; only 6 claims carry the author's section mapping).
4. Decide the treatment of H25 (not supported): state the absolute out-of-time ECE (.018–.039) rather than a ratio, and say that AUC is not consistently lower out of time for this configuration.
5. H20/H23: state the vintage pattern without a cause (no further test has been run).
6. Table T7a is 12 numeric columns wide; split or rotate for the journal template. T1–T15 LaTeX files are bare `tabular` environments (no caption/label; captions are in the manifest).
7. H1–H4 were not frozen before running (stated in the task prompts, first committed with results); the paper should say so.
8. Rerun `scripts/reproduce_all.sh` on a clean checkout before submission (the commands marked [inferred] were never re-run in this audit).
9. Placeholder title: "Calibration, Base Rates and Decision Thresholds in Credit-Risk Classification: A Multi-Dataset Study" (authors decide).

## Decisions log
| Date | Decision | Reason |
|------|----------|--------|
| 2026-10-14 | Work on branch `journal-extension`; do not modify `app.py`, templates, or the CSV | Keep baseline reproducible; new code goes in new files |
| 2026-10-14 | Use `/Library/Developer/CommandLineTools/usr/bin/git` (default `git` blocked by Xcode license prompt) | Avoid needing sudo |
| 2026-10-14 | Q1: treat pandas 3.0.2 in `venv/` as the baseline env; legacy fill bug replicated only via `legacy_fill_bug=True`; also run properly-imputed variant; never inplace-fillna on a selection in new code | Table I was produced under the bug |
| 2026-10-14 | Q2: full CSV cached once to `.cache/lc_full.parquet` (gitignored; `.gitignore` edited by append only, not committed); dev data = seeded 200k sample, drawn from *labelled* rows (after label filter) so D2 has exactly 200k modelled rows; full 2.26M for out-of-time/final runs | Compute + reproducibility |
| 2026-10-14 | Q3: headline label = Fully Paid vs Charged Off/Default; legacy 5-status label only for reproduction + sensitivity; report both | Remove right-censored/noisy labels |
| 2026-10-14 | Q4: headline features = application-time incl. `interest_rate`, with `fico_mean` instead of grade-derived `credit_score`; endogenous-feature switches (`use_interest_rate`, `credit_score` in {fico, grade_derived, none}) implemented, ablation run in Phase 3 | Avoid grade/rate leakage in the headline |
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
| 2026-10-16 | P3D answers folded into tasks: H15–H19 pre-registered first (6fa9d00); "not equivalent" ≠ "worse" and TOST-fail with CI∋0 = INCONCLUSIVE; EM diagnostic precedes any covariate-shift wording; full 128k EBM folds; Optuna 25 trials kept (wall-clock ≈1 h 15 min < 3 h cap; sequential estimate 3.9 h CPU); leakage ablation renamed endogenous-feature ablation | As specified |
| 2026-10-16 | Terminology: "leakage ablation" → "endogenous-feature ablation" (interest_rate is known at origination but is the lender's own risk price). Renamed `experiments/leakage_ablation.py` → `experiments/endogenous_feature_ablation.py`; updated `creditrisk/config.py`, `creditrisk/repeats.py` docstrings and tracker lines for Phase 1 config note, Phase 3 item, Q4 decision. **Left unchanged (frozen/historical):** pre-registration rows H10 and "Other fixed choices" in the Phase 3C block (commit 158e212) and the 2026-10-16 P3C decision-log entry, which still say "leakage" | Pre-registered wording is frozen |
| 2026-10-16 | P3C answers folded into tasks: .gitignore negation rules (no force-add); H9–H14 pre-registered first (158e212) with ±.005 AUC / ±2% ROI margins; interest-rate-only ranking = higher rate first (lower first as baseline); leakage cells A–F with E flagged as partial removal; drift fixes incl. oracle excluded from H14 | As specified |
| 2026-10-16 | Infrastructure: `repeats.get_dataset` accepts ablation cells (`D3:ir0:csgd`), `creditrisk/capacity.py` (count/dollar approval + ROI), `experiments/oot_scores.py` (cached OOT scores, 10 models seed 42 + 5 models seeds 43/44) | Shared by stats, capacity v2, drift |
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
| 2026-10-17 | Phase 4 pre-registered (H20–H22) at b3240f4 before tasks A–C | Same discipline as 3B–3D |
| 2026-10-17 | Q1/Q6: term-and-vintage split first; H20 supported (term mix does not explain the sign change), so SHAP-by-vintage was NOT run | Per instruction, run only if A did not explain the sign change — it did not, so the by-vintage SHAP is left as an open question (see below) |
| 2026-10-17 | Q2: SHAP on equivalent boosters (LightGBM, XGBoost, CatBoost, GB) + tuned RF, LR, WoE scorecard, EBM; 2,000 rows × 3 repeats; ranks only | Scales differ (RF probability scale) |
| 2026-10-17 | Q3: reason codes from the EBM, reported with and without interest_rate | Lender's own price is not borrower-actionable |
| 2026-10-17 | Q4: Phase 5 deployed model = calibrated tuned LightGBM, EBM as interpretable companion; the paper names no winner | LightGBM/CatBoost/EBM statistically tied (H9, H18) |
| 2026-10-17 | Q5: primary results stay at default hyperparameters (as pre-registered); tuned results go in a robustness table | Tuning gain ≤.0023 AUC for boosters (H16) |
| 2026-10-17 | CORRECTION to the Phase 4 pre-registration's "Fixed design choices": it says EBM "interactions=0 as in 3D"; the 3D EBM actually used interactions=10. H20–H22 hypotheses and decision rules are unchanged. Phase 4 used the 3D configuration (interactions=10; each interaction term split equally between its two features) as primary and reports interactions=0 as a sensitivity (H22 .989 vs .989 at k=3) | Keep the explanation model identical to the model that was benchmarked; frozen text left unedited |
| 2026-10-17 | SHAP "without rate" reported for both cell D (rate out, FICO kept) and cell F (rate and score out); the prompt asked for A vs F | Cell F removes the score as well; D isolates the rate |
| 2026-10-17 | Phase 5 pre-registered (H23–H24) at f60b319; branch `app-v2` created from journal-extension after 5A; `paper-v1-baseline` untouched | Per instruction; app/template/Docker work only on app-v2 |
| 2026-10-17 | Q1: 36-month refit run first (H23 supported) | Cheap; removes the training-composition confound |
| 2026-10-17 | Q2: output = calibrated PD + three review tiers anchored on π (0.75π, 1.5π) | Approve-all is not beaten by any cutoff on mature vintages |
| 2026-10-17 | Q3: isotonic default, Platt via `--calibrator platt`, annual recalibration documented | Isotonic ≈ Platt on ECE; drift dominates |
| 2026-10-17 | Q4: direction-aware labels (value vs training median, category-specific, explicit missing); interest_rate in internal codes only, borrower view hides it unless `BORROWER_SHOW_RATE=true` | Lender's own price is not borrower-actionable |
| 2026-10-17 | Q5: app.py/templates edited on app-v2 only | Baseline stays reproducible at the tag |
| 2026-10-17 | Q6: tests for creditrisk/ and Flask routes; Dockerfile for the app (not built: Docker absent) | |
| 2026-10-17 | Interpretation: "D3 200k sample plus the full mature training split" → the full 667k mature population (it contains the 200k sample), 80/20 stratified split | The sample is a subset; 'plus' would double-count |
| 2026-10-17 | Deployment trained on 80% (20% held out for the model card), EBM interactions=10 as in 3D | Reportable held-out metrics |
| 2026-10-17 | Required inputs: no defaults or zero-filling; `employment_length` accepts explicit "not reported" (the model has a missing-indicator feature); `emp_length_missing` is derived, not an input; legacy `revolving_utilization`/`credit_score` fields are gone (the study's feature set uses fico_mean and no revolving utilisation) | Serve exactly the features that were trained and benchmarked |
| 2026-10-17 | Bundle (1.38 MB, under the 5 MB limit) is committed on app-v2 via negation rules `!models/deploy/bundle.joblib` and `!models/deploy/*.csv` in `.gitignore` (no force-add); the legacy `models/*.pkl` stay ignored | Small enough to version; regenerated deterministically by `scripts/train_deploy.py` |
| 2026-10-17 | Phase 6 pre-registration (H25) at e647452, before task C; tracker merged into journal-extension from app-v2 (tracker file only; app code stays on app-v2) | Per instruction |
| 2026-10-17 | Q2: deployment model has its own table (T12e/T12f/T12g); deployment test metrics and tier report copied into `experiments/results/phase5_deploy_*.csv` on journal-extension | The export script must run on the study branch |
| 2026-10-17 | Q3: tables as CSV + LaTeX booktabs, figures as PNG (300 dpi) + PDF; the figures are regenerated from the result CSVs (the reliability figure from cached scores) and may differ in style from the earlier PNGs in experiments/figures | Vector output requires regeneration |
| 2026-10-17 | Q4: H25 run on the protocol folds (mixed terms for 2013/2014, 36m for 2015) with calibration on OOF scores of the training window (as deployed), not on year T−1 labels; the 36m-only variant is secondary | Matches the deployed procedure; verdict pre-registered |
| 2026-10-17 | Q5: the H20/H23 pattern is stated without a cause | The pre-registered rule |
| 2026-10-17 | Q6: placeholder title recorded; authors decide | Per instruction |
| 2026-10-17 | T15 / claims audit are parsed mechanically from the tracker's claim tables; the category mapping (WRONG = WRONG/REJECTED/WITHDRAWN/NOT SUPPORTED, etc.) is documented in `claims_audit.md` | Avoids a second hand-maintained list |
| 2026-10-17 | T14: H1–H4 carry no frozen pre-registration commit (stated in the Phase 2/3A task prompts; first committed with results at b7a4fee and 7ac7a5d) | Honest disclosure |
| 2026-10-17 | Verification corrections: H4 J-lost wording, H12 labelled as a mean over k, 3A AUC-drop range (see Phase 6 results) | Found by `paper_assets/verify.py` |

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

### Phase 6 changes to the claim table
| Claim | Status | Evidence |
|---|---|---|
| "Calibration drifts out of time, so recalibrate annually" (model card / app) | **SUPPORTED, but the size is revised** | H25 NOT SUPPORTED as a ratio claim: deployed configuration's ECE on the new year is .034/.018/.039 (6–14× the 133k in-time ECE .0028) but only 2.4/2.4/5.8× its own small in-window hold-out ECE |
| "Out-of-time AUC is about .016 lower" (model card, from Phase 3) | **WEAKENED for the deployed configuration** | AUC gap vs in-window hold-out +.023/+.006/−.012; no consistent drop |
| "J lost to a stale threshold is at most .007" (Phase 3B table) | **CORRECTED** | fold means ≤ .004; worst single cell .0146 (6.5% of J_max) |
| "Expected-profit gain falls from $191.6k to $102.5k with dollar capacity" | **CLARIFIED** | means over k = 30–90%; at k = 50% $242.1k → $122.9k (−49%) |

### Phase 5 changes to the claim table
| Claim | Status | Evidence |
|---|---|---|
| "Default-risk ranking loses to random under capacity constraints" | **WEAKENED, vintage-dependent; no cause established** | H23 supported: restricting train, calibration and test to 36-month loans keeps 2013 (−$21k) below 2014 (+$12k) and 2015 (+$54k) |
| "Youden / 0.5 cutoff and fixed 15/30/50/70% bands classify applicants" (conference app) | **REPLACED** | app v2 reports calibrated PD and three review tiers anchored on π; tiers are not accept/reject cutoffs |
| "A deployed model can reproduce the study's predictions" | **SUPPORTED** (H24) | max |Δ| 1.1e-16 reload, 0.0 API vs library, 1,000 loans |
| In-time reliability per tier | **HOLDS in-time only** | observed − PD within CI in all 3 tiers on the held-out split |

### Limitations added in Phase 5
- **Prototype status:** not validated for credit decisions; no fairness, stability, security or regulatory review; Docker artefacts were written but not built.
- **Funded loans only; mature vintages 2010–2015 only** (no 2016+ loans, no 60-month loans after 2013). Out-of-time performance is worse (AUC ≈ −.016; ECE ≈ 8–13× in-time), hence annual recalibration.
- **Tiers are review tiers, not accept/reject cutoffs** (approve-all is not beaten by any threshold rule on mature vintages).
- **Held-out test is a random in-time split;** the reliability-per-tier check does not cover drift.
- **Input policy:** 60 of ~1,060 scanned held-out loans (5.7%) failed validation (missing or out-of-range values), so the parity sample is the first 1,000 valid ones; the app rejects such inputs rather than imputing.
- **Reason-code wording** is descriptive relative to the training median, not causal; interest_rate hidden from borrower-facing codes by default (`BORROWER_SHOW_RATE`), still present in the internal `reason_codes` field; legal adequacy is out of scope.
- **Model trained on 80%** of the mature population (20% held out for the card); a production refit on 100% would change the numbers slightly.

### Phase 4 changes to the claim table
| Claim | Status | Evidence |
|---|---|---|
| "Default-risk ranking earns less than random under capacity constraints" | **WEAKENED further — vintage-dependent, not a term-mix effect** | 36m only (H20): 2013 −$33k, 2014 +$3k (n.s.), 2015 +$54k at k=50%; all-term 2013 −$106k, 2014 +$53k; 60m loans in 2014 +$228k. Cause untested (hypothesis: price-for-risk margin compression 2013→2015; model training composition also differs by fold) |
| "SHAP: interest rate, credit score and DTI are the top 3 features" | **WEAKENED** (interest rate first CONFIRMED) | interest_rate first in 8/8 models (29–38%); fico_mean top-3 only in LR; annual_income and loan_amount fill ranks 2–3 for boosters; DTI top-3 for CatBoost, RF, WoE |
| "Feature attributions are similar across models" | **CONFIRMED** for ranks (H21) | Spearman ≥.83 all pairs, ≥.96 among boosters; dominated by one feature |
| "EBM/interpretable model can give consistent explanations" (new) | **SUPPORTED** (H22) | top-3 reason codes overlap ≥2/3 with tuned LightGBM for 98.9% of loans (rate allowed), 98.7% (excluded); not a regulatory-compliance claim |

### Phase 3D changes to the claim table
| Claim | Status | Evidence |
|---|---|---|
| "Default-risk ranking earns less than random under capacity constraints" (3B/3C) | **WEAKENED — vintage-dependent** | holds on D3 in-time and in OOT 2013 (−$106k at k=50%); reverses in 2014 (+$53k) and 2015 36m (+$54k, CI excludes 0), where it beats expected-profit ranking at k=50–70% |
| "Resampling is a viable alternative to threshold moving" | **WEAKENED** | undersampling matches threshold moving (F1 equal; RF +.004 and +.0065 AUC); SMOTE does not move boosted-tree probabilities and costs −.009…−.018 AUC for RF/LightGBM/XGBoost; no resampling beats unweighted + Youden |
| "Hyperparameter tuning changes model ranking" | **CONFIRMED for RF only** | RF +.022 AUC (default RF is outlier); boosters +.0005…+.0023; tuned spread .0068 vs .0254 |
| "Ensembling (voting/stacking) improves credit-risk discrimination" | **REJECTED** | +.0002/+.0007 AUC, equivalent to best single (H17) |
| "Interpretable models are competitive" | **SPLIT** | EBM equivalent to the best booster (−.0009); WoE scorecard −.0085 AUC with rate, −.0167 without (not equivalent) |
| "Results depend on the 200k development sample" | **REJECTED** (H19) | 667k: calibration, weighting and AUC conclusions unchanged; AUC +.004…+.007 |
| "EM prior-shift correction fails because of covariate shift" (3C wording) | **WITHDRAWN** | EM works within a period (|π̂−π| ≤.013); failure only across years; calibration drift, not separable from covariate/concept drift |

### Phase 3C changes to the claim table
| Claim | Status | Evidence |
|---|---|---|
| "CatBoost is the best model" (Sec IV-B/V) | **REJECTED** as robust | nominal best on D1 and OOT only; equivalent set of 3–4 models in every in-time dataset; OOT lead ≤.004 over LightGBM/XGBoost except 2013 |
| "No clear winner among modern GBDTs" | **CONFIRMED** | H9: 4/5 boosting models equivalent to best on D3 (TOST ±.005, N–B, Holm) |
| "Ranking by default risk helps capacity-constrained lending" | **REJECTED** (count and dollar capacity, profit and ROI) | 10/10 models below random on ROI at k=50%; gain −$41k per 1000 (dollar) |
| "The model adds value beyond the lender's own rate" | **WEAKENED** | rate-only (high first) captures .0133 of the .0233 ROI gain of the best ranking; but still significantly below it (ΔROI −.0092, CI excl. 0) |
| "Count-based capacity results generalise to dollars" | **WEAKENED** | expected-profit gain falls from $191.6k to $102.5k per 1000 (−46%); *[Phase 6 clarification: these are means over capacity levels k = 30–90%; at k = 50% the figures are $242.1k and $122.9k]* |
| "t*≈π does not depend on feature set" | **CONFIRMED** (H10) | max mean \|t*−π\| .012 across 6 cells |
| "interest_rate drives performance and the capacity paradox" | **CONFIRMED** (H11) | ΔAUC .048; removing the rate flips the default-risk capacity gain from −$39k to +$30k/+$49k |
| "Recalibrating on recent data fixes drift" | **REJECTED** (H14) | only trailing-6-month isotonic helps (1 of 3 folds ≥30%); prior-shift EM fails when calibration and EM batch come from different years (3D diagnostic: works within a period) |

### Phase 3B changes to the claim table (supersedes Phase 2 profit magnitudes)
| Claim | Status | Evidence |
|---|---|---|
| Phase 2 profit magnitudes (profit-max accepts ~87%, +$162k over Youden) | **SUPERSEDED** | censored-vintage artefact; on D3 profit-max accepts ≥99.8% at ≤5% and equals approve-all |
| "Youden gives the best operating point" | **CHANGED (stronger)** | D3 3%: Youden earns 54% less than approve-all ($410k vs $763k per 1000; CI excludes 0); no cutoff beats approve-all at ≤5% |
| "Thresholds learned on older data go stale under drift" (H4) | **NOT SUPPORTED** | \|Δπ\| up to .029 but t* moves opposite (sign agreement 13%); Youden J lost to a stale threshold: fold means .0021/.0040/.0035 (≤1.6% of J_max), worst single (fold, model, seed) cell .0146 (6.5% of J_max; CatBoost/RF). *[Phase 6 correction: the earlier "≤ .007" was the seed-42 maximum]* |
| "Isotonic calibration on a recent vintage remains valid" | **WEAKENED** | ECE on next vintage .022–.036 vs .001–.003 in-sample (10–25× worse); worse than raw in 2014 |
| "Ranking by default risk is useful for capacity-constrained lending" | **CHANGED** | all 10 models earn less than random selection at k=30–70% (lower-risk loans carry lower interest); expected-profit ranking beats random by +$66k…+$266k per 1000 |
| "Model family matters little" | **CONFIRMED** for AUC (spread .027–.036) and expected-profit ranking; **WEAKENED** as a ranking (Spearman .17 across folds) | |
| Phase 2 calibration/weighting/leaf-purity results | **CONFIRMED on 3 datasets** | H8 replicates |

### Phase 3A changes to the picture
- The Phase 2 profit conclusions (profit-max accepts ~87%, +$162k over Youden, recall-.80 loses to approve-all) are **conditional on a D2 sample that is ~50% censored vintages**. On mature cells only (≤2013, or 2014/2015 36-month) approve-all is optimal, Youden loses 53%, and even the expected-profit rule is −$32k/−$51k below approve-all.
- Under 5% p.a. discounting the whole book is unprofitable (approve-all −$297 per applicant on D2; +$475/applicant on mature cells), and at 8% profit-max rejects everything on D2.

### Limitations added in Phase 4
- **Reason codes are not adverse-action notices.** Whether they satisfy adverse-action-notice requirements (ECOA/Reg B, FCRA) is a legal question outside this study; no compliance claim is made.
- **Direction-blind labels:** a positive contribution does not always mean "high value"; non-monotone features need direction-aware wording before any customer-facing use.
- **Attribution is conditional on the model and background distribution:** TreeSHAP (path-dependent), LinearExplainer (5,000-row background) and EBM term scores answer slightly different questions; the RF is on the probability scale, so only ranks and shares are compared. EBM interaction terms are split equally between two features (an arbitrary convention).
- **Attribution is not causation;** interest_rate dominance reflects the lender's own risk-based pricing.
- **H20 confound:** OOT folds 2013/2014 train on mixed terms, 2015 on 36m only; the cross-vintage contrast is therefore also a training-set contrast. The explanation for the sign change is hypothesis only.
- **2,000-row subsamples** per repeat (3 repeats): rank stability is reported, but ranks 6–16 are close together (shares 1–6%) so their order is not a finding.

### Limitations added in Phase 3D
- **EM:** the diagnostic shows the prior-shift estimator is sound within a period but cannot be used across years; we make no covariate-shift claim.
- **Tuning on a 100k subsample** (inner 3-fold, 25 trials, n_estimators in the search space, no early stopping); the nested design leaves the outer-test fold untouched, but tuned hyperparameters may differ on the full 160k; fold-level t-CIs (n=5) assume independent folds. Calibrated t* for tuned/default models uses a 20% hold-out of the outer-training part (single isotonic fit), not a 5-fold OOF.
- **SMOTE variant:** SMOTENC on ordinal nominal codes (trees) and plain SMOTE after one-hot (LR); sampling_strategy=1.0; resampled tree variants use ordinal codes (LightGBM without native categoricals), so AUC differences vs the unweighted baseline mix resampling with encoding for LightGBM.
- **EBM:** fitted on full 128k-row folds (no subsample), 10 interactions, default bins; WoE: 10 quantile bins, smoothing .5, no monotonic constraints.
- **667k is a selected mature population** (strict label, 2010–2013 any term + 2014/2015 36m); the 1.345M strict set contains censored 2016+ loans and is used only for threshold/calibration checks.
- **Quick-check capacity CIs** resample test rows only (400 Poisson resamples) and treat the 5 models × 3 seeds as exchangeable replicates.

### Limitations added in Phase 3C
- **Loan-size confound:** count-based capacity favours large loans (expected-profit ranking approves 68.7% of dollars at 50% of loans); dollar capacity and ROI are reported alongside.
- **interest_rate is the lender's own risk price**, not an exogenous borrower attribute; its dominance (36% of SHAP) and the capacity paradox are properties of risk-based pricing, and removing it is an ablation, not a deployable scorecard.
- **Funded loans only:** all capacity and ROI statements are relative to the historically approved population.
- **Hindsight in drift fixes:** T−1 labels, including the trailing-6-month window, were realised after year T−1 ended; EM needs unlabeled test scores of the full year T batch.
- **N–B tests** treat the 10 repeats as resamples of one fixed 200k sample (no population-level variance); OOT per-fold tests resample test rows only.
- **Equivalence margins** (±.005 AUC, ±2% of approve-all ROI) are conventions, not decision-theoretic; H13(b)'s ±.0012 ROI margin is very tight.

### Limitations added in Phase 3B
- **D3 is a selected mature population** (strict label, 2010–2015 vintages, 36m-only for 2014–2015): results speak to loans of that vintage/term mix, not to 60-month loans of 2014+ or to recent vintages.
- **Funded loans only:** "approve-all is optimal" is relative to the historically approved population; rejected applicants are unobserved.
- **Discounting is an illustrative hurdle** (grid 0/2/3/5/8%), applied with uniform-payment timing from totals; cash-flow timing and collection costs beyond `collection_recovery_fee` are not observed.
- **Hindsight labels** in the T−1 calibration/threshold sets; OOT folds have a single deterministic split and three model seeds.
- **Profit regressors** (R² .06 on total profit) are noisy; their value comes from the loan-size × rate structure, not from default prediction.

### Limitations added in Phase 3A
- **Hindsight labels:** the T−1 calibration/threshold set uses outcomes that were only observable after year T−1 (a loan issued in 2012 resolves in 2015); a live system would have fewer resolved labels.
- **Censoring:** strict labelling drops unresolved loans; late vintages are survivorship-biased and 60-month 2014–2015 cells are truncated. Headline results restricted to mature cells / vintages ≤2015.
- **Discounting approximation:** only totals are available; payments are assumed uniform between issue and last payment (recoveries on charged-off loans may arrive later). 5%/8% rates are scenario hurdles, not a funding-cost estimate.
- **Selection:** funded loans only; "approve-all optimal" means "within the historical approved population", not for rejected applicants.

## Title note (2026-10-17)
"Ensemble" is not justified as a *combination* method: voting and stacking are equivalent to the best single booster (+.0002 / +.0007 AUC, H17). It could stay only as shorthand for tree-ensemble learners, which would mislead readers. The evidence points to calibration and decision thresholds. Candidate titles:
1. *Calibration, Not Classifier Choice: Decision Thresholds for Credit-Risk Models on Lending Club Loans*
2. *Where to Cut: Calibrated Probabilities and Decision Thresholds in Credit Default Prediction*
3. *Threshold Choice and Probability Calibration in Machine-Learning Credit Scoring: A Multi-Model, Out-of-Time Evaluation*
4. *Beyond Youden: Calibration, Class Weighting and Profit-Aware Decisions for Peer-to-Peer Credit Risk*

## Paper-ready summary (10 headline findings; datasets: D1 100k legacy-label one quarter, D2 200k strict 2007–2018, D3 200k of the 667k mature-vintage population, OOT = rolling-origin 2013/2014/2015)
1. **Threshold matters more than the model family at 0.5, not after tuning.** D2, 10 repeats: balanced recall@0.5 spread across 10 models .633 vs AUC spread .027; after Youden tuning recall spread .065 (threshold noise). *CONFIRMED (0.5-vs-tuned), WEAKENED (tuned ranking).*
2. **After calibration the Youden optimum is the prevalence, and weighting is irrelevant.** mean |t*−π| .006/.005/.004 (D1/D2/D3), max .0056 on 667k; tuned-recall gap none vs balanced .005/.001/.005; raw balanced ECE .22–.25 → .008–.012. *H1, H3, H8, H19 SUPPORTED.*
3. **RF/ET "threshold failure" is leaf purity, not weighting.** D3 balanced ET recall@0.5 .006 → .631 (leaf 1 → 20); tuned RF picks min_samples_leaf ≈33. *REPLICATES.*
4. **Boosted trees are interchangeable; combining or tuning them does not help.** D3 AUC: LightGBM .6832, XGBoost/GB/CatBoost within .0011 (equivalent, H9); voting/stacking +.0002/+.0007 (H17); tuned boosters +.0005…+.0023 (RF +.0224, H16); EBM −.0009 (equivalent), WoE scorecard −.0085 (H18). *SUPPORTED.*
5. **Youden is not profit-optimal; in mature vintages approving everyone beats every threshold rule.** D3 P2-3% per 1000 applicants: approve-all $763k, Elkan $675k, F1-max $474k, Youden $410k (H5); expected-profit rule not better OOT (H6). Exception: 8% hurdle rate. *SUPPORTED.*
6. **Default-risk ranking under capacity is vintage-dependent, not a pure default-risk story.** Dollar capacity, k=50%, P2-3%: D3 in-time and OOT 2013 below random (−$106k), 2014 +$53k, 2015 36m +$54k; 36m only 2013 −$33k → 2014 +$3k → 2015 +$54k (H20); refit on 36m only −$21k → +$12k → +$54k (H23); expected-profit ranking beats random in every fold (+$22k…+$193k). *WEAKENED; cause untested.*
7. **Interest rate is the lender's own price and carries the result.** D3: removing rate and score costs .0483 AUC and flips the capacity gain −$39.1k → +$49.3k (H11); interest_rate is 29–38% of mean |attribution| and ranked first in all 8 models (H21). *SUPPORTED.*
8. **Calibration drifts across years and cheap fixes mostly fail.** OOT ECE raw .041/.022/.034; trailing-6-month isotonic helps 1 of 3 folds (H14 rejected); EM prior shift works within a period (|π̂−π| ≤.013) and fails across years. *REJECTED (fix), diagnostic only (cause).*
9. **Threshold moving matches or beats resampling.** D3, 5 repeats: undersampling ≈ threshold moving (F1 equal); SMOTE moves boosted-tree probabilities little and costs −.009…−.018 AUC for RF/LightGBM/XGBoost (H15 not supported). *WEAKENED.*
10. **Explanations agree across models, and reason codes from an interpretable model track the best booster.** Spearman of attribution ranks ≥.83 for all pairs (≥.96 among boosters, H21); EBM and tuned-LightGBM top-3 reason codes overlap ≥2/3 for 98.9% of loans (98.7% without interest rate, H22). *SUPPORTED; not a compliance claim.*

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
- [ ] (Phase 4) Refit the 2013 and 2014 OOT folds on 36-month loans only to remove the training-composition confound in H20; then (if still unexplained) SHAP by vintage and term.
- [ ] (Phase 5) Direction-aware reason-code wording and legal review before any customer-facing use.
