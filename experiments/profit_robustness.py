"""Phase 3A / task B: profit-rule robustness under P2 (discounted, net of
collection fees), with 0% / 5% / 8% sensitivity. D2 strict label, cached scores."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR, FIGURES_DIR
from creditrisk.calibration import make_calibrators, apply_cal
from creditrisk.data import full_frame, profit_p1, profit_p2
from creditrisk.repeats import get_dataset, load_score
from creditrisk.stats import summarize, paired_mean_diff_ci
from creditrisk import thresholds as T
from creditrisk.plotting import style, SERIES, MUTED, INK
import matplotlib.pyplot as plt

MODELS = ['Logistic Regression', 'LightGBM', 'XGBoost', 'CatBoost', 'Gradient Boosting']
SCEN = ['P1', 'P2_0%', 'P2_5%', 'P2_8%']
RULES = ['approve_all', 'fixed_0.5', 'youden', 'f1_max', 'recall_0.80', 'elkan', 'profit_max']


def scenario_profits(idx):
    F = full_frame().loc[idx]
    return {'P1': profit_p1(F).values, 'P2_0%': profit_p2(F, 0.0).values,
            'P2_5%': profit_p2(F, 0.05).values, 'P2_8%': profit_p2(F, 0.08).values}


def main(n_rep=10, n_boot=1000):
    X, y, _ = get_dataset('D2')
    yv = y.values
    PR = scenario_profits(X.index)
    rows, boots = [], []
    for rep in range(n_rep):
        for model in MODELS:
            z = load_score('D2', rep, model, 'none')
            tv, te = z['idx_tv'], z['idx_te']
            cal = make_calibrators(z['oof'].astype(float), yv[tv])['isotonic']
            so, st = apply_cal(cal, z['oof'].astype(float)), apply_cal(cal, z['te'].astype(float))
            y_tv, y_te = yv[tv], yv[te]
            t_y = T.youden_threshold(y_tv, so)
            base_thr = {'approve_all': np.inf, 'fixed_0.5': 0.5, 'youden': t_y,
                        'f1_max': T.f1_max_threshold(y_tv, so),
                        'recall_0.80': T.recall_target_threshold(y_tv, so, 0.80)}
            for sc in SCEN:
                p_tv, p_te = PR[sc][tv], PR[sc][te]
                c_fn, c_fp = -p_tv[y_tv == 1].mean(), p_tv[y_tv == 0].mean()
                thr = dict(base_thr)
                thr['elkan'] = T.elkan_threshold(c_fp, c_fn)
                thr['profit_max'] = T.profit_max_threshold(so, p_tv)
                for rule in RULES:
                    o = T.policy_outcome(st, thr[rule], y_te, p_te)
                    rows.append(dict(scenario=sc, rep=rep, model=model, rule=rule,
                                     threshold=thr[rule], C_FN=c_fn, C_FP=c_fp, **o))
                if rep == 0:
                    ap = {r: (st < thr[r]).astype(float) for r in RULES}
                    for r in RULES:
                        for ref in ('youden', 'approve_all'):
                            if r == ref:
                                continue
                            d = p_te * (ap[r] - ap[ref]) * 1000
                            m, lo, hi, p = paired_mean_diff_ci(d, n_boot, seed=5)
                            boots.append(dict(scenario=sc, model=model, rule=r, vs=ref,
                                              d_profit_per_1000=m, ci_lo=lo, ci_hi=hi, p=p))
    R = pd.DataFrame(rows)
    S = summarize(R, ['scenario', 'model', 'rule'],
                  ['profit_per_1000', 'accept_rate', 'bad_rate_approved', 'threshold', 'C_FN', 'C_FP'])
    S.round(4).to_csv(RESULTS_DIR / "phase3a_profit_robust.csv", index=False)
    PB = pd.DataFrame(boots); PB.round(1).to_csv(RESULTS_DIR / "phase3a_profit_robust_bootstrap.csv", index=False)
    pd.set_option('display.width', 250)
    for model in ('LightGBM',):
        t = S[S.model == model].pivot(index='rule', columns='scenario', values='profit_per_1000_mean').loc[RULES]
        sd = S[S.model == model].pivot(index='rule', columns='scenario', values='profit_per_1000_sd').loc[RULES]
        print(f"\n{model}: profit per 1000 (mean) [sd]"); print((t / 1000).round(1).to_string())
        print((sd / 1000).round(1).to_string())
        a = S[S.model == model].pivot(index='rule', columns='scenario', values='accept_rate_mean').loc[RULES]
        print("acceptance:\n", a.round(3).to_string())
    print("\nprofit_max acceptance by scenario/model:")
    print(S[S.rule == 'profit_max'].pivot(index='model', columns='scenario', values='accept_rate_mean').round(3).to_string())
    print("\nElkan costs (mean over models/reps):", S[S.rule == 'elkan'].groupby('scenario')[['C_FN_mean', 'C_FP_mean']].mean().round(0).to_dict())
    # rule ordering per scenario (mean over models)
    order = (S.groupby(['scenario', 'rule']).profit_per_1000_mean.mean().unstack(0).loc[RULES])
    print("\nmean over models, $000 / 1000 applicants:\n", (order / 1000).round(1).to_string())
    print("\nrank of rules per scenario:\n", order.rank(ascending=False).astype(int).to_string())
    print("\nrecall_0.80 - approve_all paired bootstrap (rep 0):")
    print(PB[(PB.rule == 'recall_0.80') & (PB.vs == 'approve_all')].round(0).to_string(index=False))
    print("\nprofit_max - youden (rep 0):")
    print(PB[(PB.rule == 'profit_max') & (PB.vs == 'youden')].round(0).to_string(index=False))

    # figure
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [2.2, 1]})
    ax = axes[0]
    w = 0.2
    sub = S[S.model == 'LightGBM']
    for k, sc in enumerate(SCEN):
        d = sub[sub.scenario == sc].set_index('rule').loc[RULES]
        ax.bar(np.arange(len(RULES)) + (k - 1.5) * w, d.profit_per_1000_mean / 1000, w * 0.88,
               yerr=d.profit_per_1000_sd / 1000, color=SERIES[k], label=sc.replace('_', ' '),
               error_kw=dict(lw=0.8, ecolor=INK))
    ax.set_xticks(range(len(RULES))); ax.set_xticklabels([r.replace('_', '\n') for r in RULES], fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel('profit per applicant ($000)'); ax.legend(fontsize=8, ncol=4, loc='upper left')
    ax.set_title('LightGBM (isotonic, unweighted): rule × profit definition, D2, 10 repeats')
    ax = axes[1]
    pm = S[S.rule == 'profit_max'].pivot(index='scenario', columns='model', values='accept_rate_mean').loc[SCEN]
    for k, m in enumerate(MODELS):
        ax.plot(range(len(SCEN)), pm[m] * 100, marker='o', ms=4, color=SERIES[k], label=m)
    ax.set_xticks(range(len(SCEN))); ax.set_xticklabels([s.replace('_', ' ') for s in SCEN])
    ax.set_ylabel('acceptance rate of profit-max rule (%)'); ax.legend(fontsize=7, loc='lower left')
    ax.set_title('Profit-max acceptance')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "profit_robustness.png"); plt.close(fig)


if __name__ == "__main__":
    main()
