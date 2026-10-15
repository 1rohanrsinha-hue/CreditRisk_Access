"""Phase 3B / task C (profit): D3 (mature vintages) rule study under P1 and
P2 at discount grid {0,2,3,5,8}%. Verdict on pre-registered H5."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR, FIGURES_DIR
from creditrisk.calibration import make_calibrators, apply_cal
from creditrisk.data import full_frame, profit_p1, profit_p2
from creditrisk.expected_profit import ExpectedProfit
from creditrisk.repeats import get_dataset, load_score
from creditrisk.stats import summarize, paired_mean_diff_ci
from creditrisk import thresholds as T
from creditrisk.plotting import style, SERIES, MUTED, INK
import matplotlib.pyplot as plt

DS = 'D3'
MODELS = ['Logistic Regression', 'LightGBM', 'XGBoost', 'CatBoost', 'Gradient Boosting']
RATES = [0.0, 0.02, 0.03, 0.05, 0.08]
SCEN = ['P1'] + [f'P2_{int(r*100)}%' for r in RATES]
RULES = ['approve_all', 'fixed_0.5', 'youden', 'f1_max', 'recall_0.70', 'recall_0.80', 'elkan',
         'profit_max', 'expected_profit']
HEAD = ['P1', 'P2_0%', 'P2_3%']          # scenarios H5 is stated for


def main(n_rep=10, n_boot=1000):
    X, y, _ = get_dataset(DS); yv = y.values
    F = full_frame().loc[X.index]
    PR = {'P1': profit_p1(F).values, **{f'P2_{int(r*100)}%': profit_p2(F, r).values for r in RATES}}
    rows, boots = [], []
    for rep in range(n_rep):
        # regressors per scenario are shared across classifiers (fit on train+val rows)
        z0 = load_score(DS, rep, MODELS[0], 'none'); tv, te = z0['idx_tv'], z0['idx_te']
        eps = {sc: ExpectedProfit().fit(X.iloc[tv], yv[tv], PR[sc][tv], seed=42 + rep, n_jobs=4) for sc in SCEN}
        mu = {sc: eps[sc].mu(X.iloc[te]) for sc in SCEN}
        for model in MODELS:
            z = load_score(DS, rep, model, 'none')
            cal = make_calibrators(z['oof'].astype(float), yv[tv])['isotonic']
            so, st = apply_cal(cal, z['oof'].astype(float)), apply_cal(cal, z['te'].astype(float))
            y_tv, y_te = yv[tv], yv[te]
            base = {'approve_all': np.inf, 'fixed_0.5': 0.5, 'youden': T.youden_threshold(y_tv, so),
                    'f1_max': T.f1_max_threshold(y_tv, so),
                    'recall_0.70': T.recall_target_threshold(y_tv, so, 0.70),
                    'recall_0.80': T.recall_target_threshold(y_tv, so, 0.80)}
            for sc in SCEN:
                p_tv, p_te = PR[sc][tv], PR[sc][te]
                c_fn, c_fp = -p_tv[y_tv == 1].mean(), p_tv[y_tv == 0].mean()
                thr = dict(base)
                thr['elkan'] = T.elkan_threshold(c_fp, c_fn) if c_fp > 0 and c_fn > 0 else np.nan
                thr['profit_max'] = T.profit_max_threshold(so, p_tv)
                mg, mb = mu[sc]
                appr = {r: (st < thr[r]) for r in RULES if r != 'expected_profit'}
                if np.isnan(thr['elkan']):               # costs degenerate (all repaid loans lose money)
                    appr['elkan'] = np.zeros(len(st), bool)
                appr['expected_profit'] = ((1 - st) * mg + st * mb) > 0
                for r in RULES:
                    a = appr[r]
                    rows.append(dict(scenario=sc, rep=rep, model=model, rule=r,
                                     threshold=thr.get(r, np.nan),
                                     profit_per_1000=p_te[a].sum() / len(a) * 1000,
                                     accept_rate=a.mean(),
                                     bad_rate_approved=y_te[a].mean() if a.any() else np.nan))
                if rep == 0:
                    for r in RULES:
                        for ref in ('approve_all', 'youden'):
                            if r == ref:
                                continue
                            d = p_te * (appr[r].astype(float) - appr[ref].astype(float)) * 1000
                            m, lo, hi, p = paired_mean_diff_ci(d, n_boot, seed=11)
                            boots.append(dict(scenario=sc, model=model, rule=r, vs=ref,
                                              d_profit_per_1000=m, ci_lo=lo, ci_hi=hi, p=p))
        print(f"[d3-profit] rep {rep} done", flush=True)
    R = pd.DataFrame(rows)
    S = summarize(R, ['scenario', 'model', 'rule'], ['profit_per_1000', 'accept_rate', 'bad_rate_approved', 'threshold'])
    S.round(4).to_csv(RESULTS_DIR / "phase3b_profit_d3.csv", index=False)
    PB = pd.DataFrame(boots); PB.round(1).to_csv(RESULTS_DIR / "phase3b_profit_d3_bootstrap.csv", index=False)

    pd.set_option('display.width', 250)
    for model in ('LightGBM',):
        t = S[S.model == model].pivot(index='rule', columns='scenario', values='profit_per_1000_mean').loc[RULES, SCEN] / 1000
        a = S[S.model == model].pivot(index='rule', columns='scenario', values='accept_rate_mean').loc[RULES, SCEN]
        b = S[S.model == model].pivot(index='rule', columns='scenario', values='bad_rate_approved_mean').loc[RULES, SCEN]
        print(f"\n{model}: profit $000 per 1000 applicants"); print(t.round(1).to_string())
        print("acceptance:\n", a.round(3).to_string()); print("bad rate among approved:\n", b.round(3).to_string())
    print("\nprofit_max acceptance by scenario/model:")
    print(S[S.rule == 'profit_max'].pivot(index='model', columns='scenario', values='accept_rate_mean').loc[MODELS, SCEN].round(3).to_string())
    # H5 verdict
    mean_diff = (S.pivot_table(index=['scenario', 'model'], columns='rule', values='profit_per_1000_mean')
                 .sub(S.pivot_table(index=['scenario', 'model'], columns='rule', values='profit_per_1000_mean')['approve_all'], axis=0))
    viol = []
    for _, r in PB[(PB.vs == 'approve_all') & PB.scenario.isin(HEAD)].iterrows():
        if mean_diff.loc[(r.scenario, r.model), r.rule] > 0 and r.ci_lo > 0:
            viol.append((r.scenario, r.model, r.rule, round(mean_diff.loc[(r.scenario, r.model), r.rule]), round(r.ci_lo)))
    print(f"\nH5 violations at {HEAD}: {len(viol)}"); [print("  ", v) for v in viol]
    # rules that beat approve-all (any scenario) at 8% / 5%
    for sc in ('P2_5%', 'P2_8%'):
        d = PB[(PB.scenario == sc) & (PB.vs == 'approve_all') & (PB.ci_lo > 0)]
        print(f"{sc}: pairs with CI lower bound > 0 vs approve-all: {len(d)}; rules: {sorted(d.rule.unique())}")
    print("\nrecall/youden vs approve-all (LightGBM, rep 0 paired CI):")
    print(PB[(PB.model == 'LightGBM') & (PB.vs == 'approve_all') & PB.rule.isin(['youden', 'recall_0.80', 'profit_max', 'expected_profit', 'elkan'])].round(0).to_string(index=False))

    # figure
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [2.2, 1]})
    ax = axes[0]; show = ['P1', 'P2_0%', 'P2_3%', 'P2_5%', 'P2_8%']; w = 0.16
    sub = S[S.model == 'LightGBM']
    for k, sc in enumerate(show):
        d = sub[sub.scenario == sc].set_index('rule').loc[RULES]
        ax.bar(np.arange(len(RULES)) + (k - 2) * w, d.profit_per_1000_mean / 1000, w * 0.88,
               yerr=d.profit_per_1000_sd / 1000, color=SERIES[k], label=sc.replace('_', ' '), error_kw=dict(lw=0.8, ecolor=INK))
    ax.set_xticks(range(len(RULES))); ax.set_xticklabels([r.replace('_', '\n') for r in RULES], fontsize=7.5)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylabel('profit per applicant ($000)')
    ax.legend(fontsize=8, ncol=5, loc='upper right')
    ax.set_title('D3 (mature vintages), LightGBM: rule × profit definition, 10 repeats')
    ax = axes[1]
    pm = S[S.rule == 'profit_max'].pivot(index='scenario', columns='model', values='accept_rate_mean').loc[SCEN]
    for k, m in enumerate(MODELS):
        ax.plot(range(len(SCEN)), pm[m] * 100, marker='o', ms=4, color=SERIES[k], label=m)
    ax.set_xticks(range(len(SCEN))); ax.set_xticklabels([s.replace('_', ' ') for s in SCEN], fontsize=8)
    ax.set_ylabel('acceptance of profit-max rule (%)'); ax.legend(fontsize=7, loc='lower left')
    ax.set_title('Profit-max acceptance by discount rate')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "profit_d3.png"); plt.close(fig)


if __name__ == "__main__":
    main()
