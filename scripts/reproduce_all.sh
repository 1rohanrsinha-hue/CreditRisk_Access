#!/usr/bin/env bash
# Regenerate every cache and result of the journal extension from the raw Lending Club CSV.
#
#   bash scripts/reproduce_all.sh --dry-run        # print the plan, run nothing
#   bash scripts/reproduce_all.sh                  # run all stages (hours; resumable)
#   bash scripts/reproduce_all.sh --from 12        # start at stage 12
#   FORCE=1 bash scripts/reproduce_all.sh          # ignore stage-done markers
#
# DATA AVAILABILITY. The Lending Club accepted-loans file `accepted_2007_to_2018Q4.csv` is NOT redistributed in this
# repository. Obtain it from its public source and place it at the repository root.
#   Source / URL / licence / date accessed:  <<< UNKNOWN (no URL or licence is recorded; see docs/data_cards/lending_club.md) >>>
# If you only have the .csv.gz, run `python extract_dataset.py` first.
#
# ENVIRONMENT. Python 3.13 (results were produced with 3.13.5 on macOS arm64, 10 cores) and the pinned packages in
# requirements-lock.txt:  python3.13 -m venv venv && venv/bin/python -m pip install -r requirements-lock.txt
# (pandas 3.0.2 matters: the conference code's pandas-3 fillna behaviour is replicated on purpose, see the tracker).
#
# RESUMABILITY. Heavy scoring stages write one checkpoint per unit (.cache/scores, .cache/oot_scores, .cache/shap,
# .cache/deploy_oot, .cache/optuna) and skip units that exist; each stage also leaves a marker in .cache/repro_done/.
# Runtimes are APPROXIMATE (10-core laptop); where the original timing was not recorded the entry says 'unrecorded, guess'.
# Commands marked [inferred] were reconstructed from each script's CLI, not re-run during the Phase 6 audit; the
# scripts of Phases 4-6 were run exactly as written here.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-venv/bin/python}; LOG=.cache/logs; DONE=.cache/repro_done
DRY=0; FROM=1
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; --from) FROM=$2; shift;; *) echo "unknown arg $1"; exit 2;; esac; shift; done
mkdir -p "$LOG" "$DONE" experiments/results experiments/figures
N=0
stage() {  # stage <id-name> <approx runtime> <command...>
  N=$((N+1)); local name=$1 rt=$2; shift 2
  if [ "$N" -lt "$FROM" ]; then return; fi
  case "$rt" in [0-9]*) rt="~$rt";; esac; printf '[%02d] %-34s %s\n' "$N" "$name" "$rt"
  if [ "$DRY" = 1 ]; then printf '       %s\n' "$*"; return; fi
  if [ -f "$DONE/$name" ] && [ "${FORCE:-0}" != 1 ]; then echo "     (done earlier; FORCE=1 to rerun)"; return; fi
  bash -c "$*" > "$LOG/repro_$name.log" 2>&1 && touch "$DONE/$name" || { echo "FAILED: $name (see $LOG/repro_$name.log)"; exit 1; }
}
if [ "$DRY" = 0 ]; then { date; uname -a; $PY --version; $PY -m pip freeze | head -200; } > "$LOG/repro_env.txt" 2>&1; fi

# ---- data
stage cache_parquet            "5 min"      "$PY -c 'from creditrisk.data import build_cache; build_cache()'"
stage d3_population            "1 min"      "$PY experiments/d3_population.py"
# ---- Phase 1
stage p1_baseline_legacy       "3 min"      "$PY experiments/baseline_repro.py --variant legacy_fill_bug"
stage p1_baseline_proper       "3 min"      "$PY experiments/baseline_repro.py --variant proper_impute"
stage p1_column_order          "unrecorded, guess 20 min"  "$PY experiments/column_order_sensitivity.py"
stage p1_protocols_D1          "unrecorded, guess 30 min"  "$PY experiments/protocol_comparison.py --dataset D1   # [inferred]"
stage p1_protocols_D2          "unrecorded, guess 1 h"     "$PY experiments/protocol_comparison.py --dataset D2   # [inferred]"
# ---- Phase 2 (weighting x threshold, calibration, profit); score stages are resumable and may be parallelised with --worker/--n-workers
stage p2_score_D1              "unrecorded, guess 1 h"     "$PY experiments/weighting_ablation.py --stage score --dataset D1   # [inferred]"
stage p2_score_D2              "unrecorded, guess 4 h"     "$PY experiments/weighting_ablation.py --stage score --dataset D2   # [inferred]"
stage p2_analyze_D1_D2         "5 min"      "$PY experiments/weighting_ablation.py --stage analyze --dataset D1 && $PY experiments/weighting_ablation.py --stage analyze --dataset D2   # [inferred]"
stage p2_leaf_sweep_D2         "unrecorded, guess 30 min"  "$PY experiments/weighting_ablation.py --stage leaf --dataset D2 --leaf-sweep   # [inferred]"
stage p2_calibration_D1_D2     "unrecorded, guess 20 min"  "$PY experiments/calibration_study.py --dataset D1 && $PY experiments/calibration_study.py --dataset D2   # [inferred]"
stage p2_profit                "unrecorded, guess 20 min"  "$PY experiments/profit_study.py   # [inferred]"
# ---- Phase 3A
stage p3a_vintage_audit        "2 min"      "$PY experiments/vintage_audit.py"
stage p3a_profit_robustness    "unrecorded, guess 10 min"  "$PY experiments/profit_robustness.py"
stage p3a_expected_profit      "unrecorded, guess 20 min"  "$PY experiments/expected_profit_study.py"
stage p3a_mature_subset        "unrecorded, guess 5 min"   "$PY experiments/mature_subset_check.py"
# ---- Phase 3B (D3)
stage p3b_score_D3             "unrecorded, guess 3 h"     "$PY experiments/weighting_ablation.py --stage score --dataset D3 --leaf-sweep   # [inferred]"
stage p3b_analyze_D3           "10 min"     "$PY experiments/weighting_ablation.py --stage analyze --dataset D3 && $PY experiments/weighting_ablation.py --stage leaf --dataset D3 && $PY experiments/calibration_study.py --dataset D3 && $PY experiments/replication_table.py && $PY experiments/profit_d3.py && $PY experiments/capacity_study.py   # [inferred]"
stage p3b_oot                  "unrecorded, guess 1 h"     "$PY experiments/oot_study.py --seeds 42 43 44 --tag phase3b_oot   # [inferred]"
stage p3b_expprofit_oot        "unrecorded, guess 30 min"  "$PY experiments/expprofit_oot.py"
# ---- Phase 3C
stage p3c_oot_scores           "unrecorded, guess 1 h"     "$PY experiments/oot_scores.py --n-jobs 4"
stage p3c_stats                "5 min"      "$PY experiments/stats_study.py"
stage p3c_capacity_v2          "10 min"     "$PY experiments/capacity_v2.py"
stage p3c_ablation             "unrecorded, guess 3 h"     "$PY experiments/endogenous_feature_ablation.py --stage score && $PY experiments/endogenous_feature_ablation.py --stage analyze"
stage p3c_drift                "10 min"     "$PY experiments/drift_fixes.py"
# ---- Phase 3D
stage p3d_quick_checks         "10 min"     "$PY experiments/quick_checks_3d.py"
stage p3d_ensembles            "5 min"      "$PY experiments/ensemble_study.py"
stage p3d_interpretable        "unrecorded, guess 1.5 h"   "$PY experiments/interpretable_baselines.py --stage score && $PY experiments/interpretable_baselines.py --stage analyze"
stage p3d_resampling           "unrecorded, guess 2 h"     "$PY experiments/resampling_study.py --stage score && $PY experiments/resampling_study.py --stage analyze"
stage p3d_scale                "unrecorded, guess 10 min"  "$PY experiments/scale_check.py --stage score && $PY experiments/scale_check.py --stage analyze"
stage p3d_optuna               "~1.25 h wall (5 processes; ~3.9 CPU-h)" "for m in 'Logistic Regression' LightGBM XGBoost 'Random Forest' CatBoost; do $PY experiments/optuna_nested.py --stage run --model \"\$m\" --trials 25 --folds 0 1 2 3 4 --n-jobs 2 & done; wait; $PY experiments/optuna_nested.py --stage collect   # [inferred: run the five models as parallel processes]"
# ---- Phase 4
stage p4_term_vintage          "~15 min"    "$PY experiments/term_vintage_check.py"
stage p4_shap                  "~10 min (3 reps in parallel) + 4 min cell F" "for r in 0 1 2; do $PY experiments/shap_study.py --stage run --reps \$r --n-jobs 2 & done; wait; for r in 0 1 2; do $PY experiments/shap_study.py --stage cellF --reps \$r --n-jobs 2 & done; wait; $PY experiments/shap_study.py --stage collect"
stage p4_reason_codes          "1 min"      "$PY experiments/reason_codes_study.py"
# ---- Phase 5 / 6 (study side, journal-extension)
stage p5_term36                "~10 min"    "$PY experiments/term36_refit.py"
stage p6_deploy_oot            "~10 min (3 folds in parallel)" "for T in 2013 2014 2015; do $PY experiments/deploy_oot.py --stage run --T \$T --n-jobs 3 & done; wait; $PY experiments/deploy_oot.py --stage collect"
# ---- Phase 7 external datasets (German Credit and Taiwan from UCI; Give Me Some Credit is not used). Data URL/licence/date: see (see docs/data_cards/)
stage p7_fetch_external        "~1 min"     "bash scripts/fetch_external.sh && $PY -m pip install -q xlrd && $PY experiments/external_cards.py"
stage p7_score_german          "~5 min"     "$PY experiments/external_study.py --stage score --dataset GERMAN --n-jobs 2"
stage p7_score_taiwan          "~60 min (2 workers)" "for w in 0 1; do $PY experiments/external_study.py --stage score --dataset TAIWAN --worker \$w --n-workers 2 --n-jobs 2 & done; wait"
stage p7_analyze_external      "~10 min"    "$PY experiments/external_study.py --stage analyze --dataset GERMAN && $PY experiments/external_study.py --stage analyze --dataset TAIWAN"
# ---- Phase 5 deployment (branch app-v2 only): `git checkout app-v2`, then
#      $PY scripts/train_deploy.py   (~8 min; writes models/deploy/)   and   $PY -m pytest   (39 tests, ~35 s)
# ---- Paper assets (journal-extension)
stage paper_assets             "~1 min"     "$PY experiments/export_paper_assets.py && $PY paper_assets/verify.py"
echo "plan complete ($N stages)"
