"""Phase 3A / supplementary: re-evaluate the Phase 2 D2 profit rules on MATURE
vintage x term cells only (issue year <= 2013, or 2014/2015 36-month), using the
cached D2 scores. Thresholds are re-picked on the mature OOF rows. Diagnoses the
censoring bias of 2016+ strict-labelled loans in the random D2 sample."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR
from creditrisk.calibration import make_calibrators, apply_cal
from creditrisk.data import full_frame, issue_year, term_months, profit_p1, profit_p2
from creditrisk.repeats import get_dataset, load_score
from creditrisk.stats import summarize, paired_mean_diff_ci
from creditrisk import thresholds as T

MODELS = ['Logistic Regression', 'LightGBM', 'XGBoost', 'CatBoost', 'Gradient Boosting']
RULES = ['approve_all', 'fixed_0.5', 'youden', 'f1_max', 'recall_0.80', 'elkan', 'profit_max']

X, y, _ = get_dataset('D2'); yv = y.values
F = full_frame().loc[X.index]
yr, tm = issue_year(F).values, term_months(F).values
mature = (yr <= 2013) | ((yr <= 2015) & (tm == 36))
print(f"D2 rows: {len(yv):,}; mature-cell rows: {mature.sum():,} ({mature.mean():.1%}); "
      f"default rate all {yv.mean():.3f} vs mature {yv[mature].mean():.3f}")
PR = {'P1': profit_p1(F).values, 'P2_5%': profit_p2(F, 0.05).values}
print("mean profit per loan  all vs mature:", {k: (round(v.mean()), round(v[mature].mean())) for k, v in PR.items()})
rows = []
for rep in range(10):
    for model in MODELS:
        z = load_score('D2', rep, model, 'none'); tv, te = z['idx_tv'], z['idx_te']
        cal = make_calibrators(z['oof'].astype(float), yv[tv])['isotonic']
        so, st = apply_cal(cal, z['oof'].astype(float)), apply_cal(cal, z['te'].astype(float))
        mt, me = mature[tv], mature[te]
        y_tv, y_te, s_tv, s_te = yv[tv][mt], yv[te][me], so[mt], st[me]
        for sc, P in PR.items():
            p_tv, p_te = P[tv][mt], P[te][me]
            c_fn, c_fp = -p_tv[y_tv == 1].mean(), p_tv[y_tv == 0].mean()
            thr = {'approve_all': np.inf, 'fixed_0.5': 0.5, 'youden': T.youden_threshold(y_tv, s_tv),
                   'f1_max': T.f1_max_threshold(y_tv, s_tv),
                   'recall_0.80': T.recall_target_threshold(y_tv, s_tv, 0.80),
                   'elkan': T.elkan_threshold(c_fp, c_fn), 'profit_max': T.profit_max_threshold(s_tv, p_tv)}
            for r in RULES:
                o = T.policy_outcome(s_te, thr[r], y_te, p_te)
                rows.append(dict(scenario=sc, rep=rep, model=model, rule=r, threshold=thr[r], **o))
R = pd.DataFrame(rows)
S = summarize(R, ['scenario', 'model', 'rule'], ['profit_per_1000', 'accept_rate', 'bad_rate_approved', 'threshold'])
S.round(4).to_csv(RESULTS_DIR / "phase3a_mature_subset.csv", index=False)
pd.set_option('display.width', 220)
for sc in ('P1', 'P2_5%'):
    t = S[(S.scenario == sc) & (S.model == 'LightGBM')].set_index('rule').loc[RULES]
    print(f"\n{sc} LightGBM, mature cells only:")
    print(t[['profit_per_1000_mean', 'profit_per_1000_sd', 'accept_rate_mean', 'bad_rate_approved_mean', 'threshold_mean']].round(3).to_string())

# ---- expected-profit rule on mature cells (regressors trained on mature train+val rows only)
from creditrisk.expected_profit import ExpectedProfit
rows = []
for rep in range(10):
    z = load_score('D2', rep, 'LightGBM', 'none'); tv, te = z['idx_tv'], z['idx_te']
    cal = make_calibrators(z['oof'].astype(float), yv[tv])['isotonic']
    so, st = apply_cal(cal, z['oof'].astype(float)), apply_cal(cal, z['te'].astype(float))
    mt, me = mature[tv], mature[te]
    for sc, P in PR.items():
        ep = ExpectedProfit().fit(X.iloc[tv[mt]], yv[tv][mt], P[tv][mt], seed=42 + rep, n_jobs=4)
        E = ep.expected(X.iloc[te[me]], st[me])
        p_te, y_te = P[te][me], yv[te][me]
        for rule, a in (('expected_profit', E > 0), ('approve_all', np.ones(len(E), bool))):
            rows.append(dict(scenario=sc, rep=rep, rule=rule, profit_per_1000=p_te[a].sum() / len(a) * 1000,
                             accept_rate=a.mean(), bad_rate_approved=y_te[a].mean()))
EP = summarize(pd.DataFrame(rows), ['scenario', 'rule'], ['profit_per_1000', 'accept_rate', 'bad_rate_approved'])
EP.round(4).to_csv(RESULTS_DIR / "phase3a_mature_expprofit.csv", index=False)
print("\nExpected-profit rule on mature cells:\n", EP.round(3).to_string(index=False))
