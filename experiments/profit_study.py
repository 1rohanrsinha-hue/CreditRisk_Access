"""Phase 2 / task E: threshold rules and realized-profit evaluation (D2, strict label).

Policy: approve a loan if score < t. realized_profit = total_pymnt - funded_amnt
(total_pymnt already contains recoveries, verified in Phase 2 task A).
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR, FIGURES_DIR
from creditrisk.calibration import make_calibrators, apply_cal
from creditrisk.models import MODEL_NAMES
from creditrisk.repeats import get_dataset, load_score
from creditrisk.stats import summarize, paired_mean_diff_ci
from creditrisk.study import analyse
from creditrisk.plotting import style, SERIES, MUTED, INK
import matplotlib.pyplot as plt

VALS = ['profit_per_1000', 'accept_rate', 'bad_rate_approved', 'Recall', 'Precision', 'F1', 'threshold']
RULE_ORDER = ['fixed_0.5', 'youden', 'f1_max', 'g_mean', 'recall_0.70', 'recall_0.80',
              'prevalence', 'elkan', 'profit_max']
FIG_MODELS = ['Logistic Regression', 'XGBoost', 'CatBoost', 'Neural Network']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    reps = list(range(a.repeats))
    X, y, profit = get_dataset("D2")
    yv, pv = y.values, profit.values
    R, C = analyse("D2", reps, with_profit=True)

    # ---- baselines per repeat
    base = []
    cost = []
    for rep in reps:
        z = load_score("D2", rep, "Logistic Regression", "none")
        te, tv = z['idx_te'], z['idx_tv']
        n = len(te)
        base.append(dict(rep=rep, approve_all=pv[te].sum() / n * 1000, reject_all=0.0,
                         oracle=pv[te][yv[te] == 0].sum() / n * 1000,
                         bad_rate_all=yv[te].mean()))
        cost.append(dict(rep=rep, C_FN=-pv[tv][yv[tv] == 1].mean(), C_FP=pv[tv][yv[tv] == 0].mean()))
    B = pd.DataFrame(base); K = pd.DataFrame(cost)
    K['elkan_t'] = K.C_FP / (K.C_FP + K.C_FN)
    B.to_csv(RESULTS_DIR / "phase2_profit_baselines.csv", index=False)
    K.to_csv(RESULTS_DIR / "phase2_profit_costs.csv", index=False)
    print("baselines (per 1000 applicants):\n", B.drop(columns='rep').agg(['mean', 'std']).round(1).to_string())
    print("data-derived Elkan costs:\n", K.drop(columns='rep').agg(['mean', 'std']).round(3).to_string())

    # ---- mean/sd over repeats for every model x weighting x calib x rule
    S = summarize(R, ['model', 'weighting', 'calib', 'rule'], VALS)
    S.to_csv(RESULTS_DIR / "phase2_profit.csv", index=False)

    # ---- paired bootstrap vs Youden, reference repeat 0 (same calib/weighting/model)
    rows = []
    z_cache = {}
    for model in MODEL_NAMES:
        for w in ('none', 'balanced'):
            z = load_score("D2", 0, model, w)
            te, tv = z['idx_te'], z['idx_tv']
            cals = make_calibrators(z['oof'].astype(float), yv[tv])
            r0 = R[(R.rep == 0) & (R.model == model) & (R.weighting == w)]
            for cn in ('raw', 'isotonic'):
                st = apply_cal(cals[cn], z['te'].astype(float))
                sub = r0[r0.calib == cn].set_index('rule').threshold
                a_y = st < sub['youden']
                for rule in RULE_ORDER:
                    if rule not in sub.index or rule == 'youden':
                        continue
                    d = pv[te] * ((st < sub[rule]).astype(float) - a_y.astype(float)) * 1000
                    m, lo, hi, p = paired_mean_diff_ci(d, a.n_boot, seed=3)
                    rows.append(dict(model=model, weighting=w, calib=cn, rule=rule,
                                     d_profit_vs_youden_per_1000=m, ci_lo=lo, ci_hi=hi, p_two_sided=p))
    PB = pd.DataFrame(rows)
    PB.to_csv(RESULTS_DIR / "phase2_profit_paired_bootstrap.csv", index=False)

    # ---- tables
    pd.set_option('display.width', 250)
    nat = S[(S.weighting == 'none') & (S.calib == 'isotonic')]
    rank = (nat[nat.rule == 'youden'].set_index('model'))
    # ranking uses Brier of the calibrated unweighted model (from cal table)
    cb = C[(C.weighting == 'none') & (C.calib == 'isotonic')].groupby('model').Brier.mean().sort_values()
    by_profit = (nat[nat.rule == 'profit_max'].set_index('model').profit_per_1000_mean
                 .sort_values(ascending=False))
    by_elkan = (nat[nat.rule == 'elkan'].set_index('model').profit_per_1000_mean.sort_values(ascending=False))
    print("Brier rank (iso, unweighted):", cb.round(4).head(5).to_dict())
    print("profit rank (profit_max):", by_profit.round(0).head(5).to_dict())
    print("profit rank (elkan):", by_elkan.round(0).head(5).to_dict())
    top = list(dict.fromkeys(list(cb.index[:3]) + list(by_profit.index[:3])))
    t = S[S.model.isin(top) & (S.weighting == 'none') & (S.calib == 'isotonic')].copy()
    t['rule'] = pd.Categorical(t.rule, RULE_ORDER, ordered=True)
    t = t.sort_values(['model', 'rule'])
    print(t[['model', 'rule', 'threshold_mean', 'profit_per_1000_mean', 'profit_per_1000_sd',
             'accept_rate_mean', 'bad_rate_approved_mean', 'Recall_mean', 'F1_mean']].round(3).to_string(index=False))
    pbt = PB[PB.model.isin(top) & (PB.weighting == 'none') & (PB.calib == 'isotonic')]
    print(pbt.round(0).to_string(index=False))

    # ---- figures (repeat 0, unweighted + isotonic-calibrated, test set)
    FIGURES_DIR.mkdir(exist_ok=True); style()
    grid = np.linspace(0.02, 0.98, 97)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.4))
    mk = {'youden': 'o', 'elkan': 's', 'profit_max': '^', 'f1_max': 'D', 'recall_0.80': 'v', 'prevalence': 'x'}
    for k, m in enumerate(FIG_MODELS):
        z = load_score("D2", 0, m, 'none'); te, tv = z['idx_te'], z['idx_tv']
        cals = make_calibrators(z['oof'].astype(float), yv[tv])
        st = apply_cal(cals['isotonic'], z['te'].astype(float))
        r0 = R[(R.rep == 0) & (R.model == m) & (R.weighting == 'none') & (R.calib == 'isotonic')].set_index('rule')
        n = len(te)
        prof = np.array([pv[te][st < t].sum() / n * 1000 for t in grid])
        ax = axes[k]
        ax.plot(grid, prof / 1000, color=SERIES[0])
        ax.axhline(B.approve_all.iloc[0] / 1000, color=MUTED, lw=0.9, ls='--')
        for j, rule in enumerate(('youden', 'f1_max', 'recall_0.80', 'elkan', 'profit_max')):
            t = r0.loc[rule, 'threshold']
            ax.plot([t], [r0.loc[rule, 'profit_per_1000'] / 1000], mk.get(rule, 'o'), ms=7,
                    color=SERIES[(j + 1) % 8], mec='white', mew=1.2, label=rule, zorder=5)
        ax.set_title(m); ax.set_xlabel('threshold t (approve if score < t)')
        # accept vs bad-rate curve
        acc = np.array([(st < t).mean() for t in grid])
        bad = np.array([yv[te][st < t].mean() if (st < t).any() else np.nan for t in grid])
        ax2.plot(acc, bad, color=SERIES[k], label=m)
    axes[0].set_ylabel('realized profit per applicant ($000)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    axes[0].text(0.97, B.approve_all.iloc[0] / 1000 - 0.04, 'approve all', fontsize=8, color=MUTED, ha='right', va='top')
    fig.suptitle('Policy profit vs threshold, D2 repeat 0 (isotonic-calibrated, unweighted); markers = rule operating points', x=0.01, ha='left', fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "profit_vs_threshold.png", bbox_inches='tight'); plt.close(fig)
    ax2.set_xlabel('acceptance rate'); ax2.set_ylabel('bad rate among approved')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_title('Acceptance vs bad rate (test, repeat 0)')
    fig2.tight_layout(); fig2.savefig(FIGURES_DIR / "accept_vs_badrate.png"); plt.close(fig2)
    print("figures written")


if __name__ == "__main__":
    main()
