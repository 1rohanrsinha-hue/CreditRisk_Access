"""Phase 3B / task D: capacity-constrained comparison (H7).
Approve the k% lowest-risk applicants by (i) each model's score (monotone in its
calibrated p; raw score used to avoid isotonic ties) and (ii) expected profit
(isotonic-calibrated p + LightGBM profit regressors). D3 x 10 repeats + OOT folds."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import RESULTS_DIR, FIGURES_DIR
from creditrisk.calibration import make_calibrators, apply_cal
from creditrisk.data import full_frame, profit_p1, profit_p2
from creditrisk.expected_profit import ExpectedProfit
from creditrisk.models import MODEL_NAMES, fit_model
from creditrisk.repeats import get_dataset, load_score
from creditrisk.stats import paired_mean_diff_ci, summarize
from creditrisk.plotting import style, SERIES, MUTED, INK
import matplotlib.pyplot as plt

KS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SCEN = ['P1', 'P2_3%']
RANKINGS = ['score', 'expected_profit']


def approve_lowest(key, k):
    n_ap = int(round(k * len(key)))
    a = np.zeros(len(key), bool)
    a[np.argsort(key, kind='stable')[:n_ap]] = True
    return a


def eval_rows(meta, keys, y, Pte, store=None):
    """keys[(ranking, model)] -> score array (lower = approve first)."""
    rows = []
    base = {sc: Pte[sc].mean() * 1000 for sc in SCEN}
    for (rk, model), (key_by_scen) in keys.items():
        for sc in SCEN:
            key = key_by_scen[sc] if isinstance(key_by_scen, dict) else key_by_scen
            for k in KS:
                a = approve_lowest(key, k)
                prof = Pte[sc][a].sum() / len(a) * 1000
                rows.append(dict(**meta, scenario=sc, ranking=rk, model=model, k=k, profit_per_1000=prof,
                                 bad_rate_approved=y[a].mean(), gain_over_random=prof - k * base[sc],
                                 approve_all=base[sc]))
                if store is not None:
                    store[(rk, sc, model, k)] = a
    return rows


def main(n_boot=1000):
    t0 = time.time()
    # ---------------- D3
    X, y, _ = get_dataset('D3'); yv = y.values
    F = full_frame().loc[X.index]
    PR = {'P1': profit_p1(F).values, 'P2_3%': profit_p2(F, 0.03).values}
    rows, store0, te0 = [], {}, None
    for rep in range(10):
        z0 = load_score('D3', rep, MODEL_NAMES[0], 'none'); tv, te = z0['idx_tv'], z0['idx_te']
        eps = {sc: ExpectedProfit().fit(X.iloc[tv], yv[tv], PR[sc][tv], seed=42 + rep, n_jobs=4) for sc in SCEN}
        mu = {sc: eps[sc].mu(X.iloc[te]) for sc in SCEN}
        keys = {}
        for model in MODEL_NAMES:
            z = load_score('D3', rep, model, 'none')
            cal = make_calibrators(z['oof'].astype(float), yv[tv])['isotonic']
            s_te = z['te'].astype(float); p_te = apply_cal(cal, s_te)
            keys[('score', model)] = s_te
            keys[('expected_profit', model)] = {sc: -((1 - p_te) * mu[sc][0] + p_te * mu[sc][1]) for sc in SCEN}
        Pte = {sc: PR[sc][te] for sc in SCEN}
        rows += eval_rows(dict(dataset='D3', fold=f'rep{rep}', rep=rep), keys, yv[te], Pte,
                          store0 if rep == 0 else None)
        if rep == 0:
            te0 = (te, Pte, yv[te])
        print(f"[capacity] D3 rep {rep} {time.time()-t0:.0f}s", flush=True)
    D = pd.DataFrame(rows)
    # ---------------- bootstrap on rep 0 (test-set sampling)
    te, Pte, y_te = te0
    boots = []
    S0 = D[D.rep == 0]
    for (rk, sc, k), g in S0.groupby(['ranking', 'scenario', 'k']):
        best = D[(D.ranking == rk) & (D.scenario == sc) & (D.k == k)].groupby('model').profit_per_1000.mean().idxmax()
        a_best = store0[(rk, sc, best, k)]
        for model in MODEL_NAMES:
            a = store0[(rk, sc, model, k)]
            lvl = paired_mean_diff_ci(Pte[sc] * a * 1000, n_boot, seed=17)
            gain = paired_mean_diff_ci(Pte[sc] * (a.astype(float) - k) * 1000, n_boot, seed=17)
            vsb = paired_mean_diff_ci(Pte[sc] * (a.astype(float) - a_best.astype(float)) * 1000, n_boot, seed=17) if model != best else (0, 0, 0, 1)
            boots.append(dict(ranking=rk, scenario=sc, k=k, model=model, best_model=best,
                              level_lo=lvl[1], level_hi=lvl[2], gain_lo=gain[1], gain_hi=gain[2],
                              d_vs_best=vsb[0], d_vs_best_lo=vsb[1], d_vs_best_hi=vsb[2]))
    B = pd.DataFrame(boots)
    # ---------------- OOT folds (10 models, seed 42)
    from oot_study import make_data, fold_indices, usable_columns, FOLDS
    Xo, yo, Po, yr, tm = make_data()
    PoS = {'P1': Po['P1'], 'P2_3%': Po['P2_3%']}
    orows = []
    for Tyr in FOLDS:
        tr, ca, te_ = fold_indices(Tyr, yr, tm)
        cols, _ = usable_columns(Xo, tr); Xc = Xo[cols]
        eps = {sc: ExpectedProfit().fit(Xc.iloc[tr], yo[tr], PoS[sc][tr], seed=42, n_jobs=4) for sc in SCEN}
        mu = {sc: eps[sc].mu(Xc.iloc[te_]) for sc in SCEN}
        keys = {}
        for model in MODEL_NAMES:
            m = fit_model(model, Xc.iloc[tr], pd.Series(yo[tr]), 'none', 42, 4)
            s_ca, s_te = m.predict_proba(Xc.iloc[ca])[:, 1], m.predict_proba(Xc.iloc[te_])[:, 1]
            cal = make_calibrators(s_ca, yo[ca])['isotonic']; p_te = apply_cal(cal, s_te)
            keys[('score', model)] = s_te
            keys[('expected_profit', model)] = {sc: -((1 - p_te) * mu[sc][0] + p_te * mu[sc][1]) for sc in SCEN}
            print(f"[capacity] OOT T={Tyr} {model} {time.time()-t0:.0f}s", flush=True)
        Pte_o = {sc: PoS[sc][te_] for sc in SCEN}
        orows += eval_rows(dict(dataset='OOT', fold=f'T{Tyr}', rep=Tyr), keys, yo[te_], Pte_o)
    O = pd.DataFrame(orows)
    A = pd.concat([D, O], ignore_index=True)
    A.round(2).to_csv(RESULTS_DIR / "phase3b_capacity.csv", index=False)
    B.round(1).to_csv(RESULTS_DIR / "phase3b_capacity_bootstrap.csv", index=False)

    # ---------------- summaries + H7
    pd.set_option('display.width', 250)
    S = summarize(D, ['scenario', 'ranking', 'model', 'k'], ['profit_per_1000', 'bad_rate_approved', 'gain_over_random'])
    S.round(2).to_csv(RESULTS_DIR / "phase3b_capacity_summary.csv", index=False)
    print("\n=== H7 (D3, ranking = score) ===")
    for sc in SCEN:
        print(f"\n[{sc}]")
        tab = []
        for k in KS:
            s = S[(S.scenario == sc) & (S.ranking == 'score') & (S.k == k)].set_index('model')
            b = B[(B.scenario == sc) & (B.ranking == 'score') & (B.k == k)]
            best = s.profit_per_1000_mean.idxmax()
            lvl = b[b.model == best].iloc[0]
            half = (lvl.level_hi - lvl.level_lo) / 2
            rng = s.profit_per_1000_mean.max() - s.profit_per_1000_mean.min()
            gmin = b.gain_lo.min()
            tab.append(dict(k=k, best=best, best_profit=s.profit_per_1000_mean.max(), range_models=rng,
                            ci_half_width_best=half, a_range_lt_halfwidth=rng < half,
                            min_gain_lo_over_models=gmin, b_all_beat_random=bool(gmin > 0),
                            approve_all=s.profit_per_1000_mean.iloc[0] * 0 + D[(D.scenario == sc)].approve_all.mean()))
        T_ = pd.DataFrame(tab); print(T_.round(1).to_string(index=False))
        print(f"clause (a): range < CI half-width at {int(T_.a_range_lt_halfwidth.sum())}/7 levels; "
              f"clause (b): all models beat random at {int(T_.b_all_beat_random.sum())}/7 levels")
    # rank stability
    for sc in SCEN:
        piv = S[(S.scenario == sc) & (S.ranking == 'score')].pivot(index='model', columns='k', values='profit_per_1000_mean')
        rs = [spearmanr(piv[a], piv[b])[0] for i, a in enumerate(KS) for b in KS[i + 1:]]
        print(f"{sc}: mean pairwise Spearman of model ranking across k = {np.mean(rs):.2f}")
        for k in (0.5,):
            r = D[(D.scenario == sc) & (D.ranking == 'score') & (D.k == k)].pivot(index='model', columns='rep', values='profit_per_1000')
            rs2 = [spearmanr(r[a], r[b])[0] for i, a in enumerate(r.columns) for b in r.columns[i + 1:]]
            print(f"{sc}: k=.5 mean pairwise Spearman across 10 repeats = {np.mean(rs2):.2f}")
        o = O[(O.scenario == sc) & (O.ranking == 'score')].pivot_table(index='model', columns=['fold', 'k'], values='profit_per_1000')
        folds = sorted(set(c[0] for c in o.columns))
        rs3 = [spearmanr(o[(f1, k)], o[(f2, k)])[0] for k in KS for i, f1 in enumerate(folds) for f2 in folds[i + 1:]]
        print(f"{sc}: mean Spearman of model ranking across OOT folds (same k) = {np.mean(rs3):.2f}")
    print("\nranking = expected_profit vs score (D3, k=.5, P2_3%), mean over repeats:")
    e = S[(S.scenario == 'P2_3%') & (S.k == 0.5)].pivot(index='model', columns='ranking', values='profit_per_1000_mean')
    print(e.round(0).to_string())
    print("\nOOT, ranking=score, P2_3%, profit by fold at k=.5:")
    print(O[(O.scenario == 'P2_3%') & (O.ranking == 'score') & (O.k == 0.5)].pivot(index='model', columns='fold', values='profit_per_1000').round(0).to_string())

    # ---------------- figure
    style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    show = ['Logistic Regression', 'LightGBM', 'XGBoost', 'CatBoost', 'Random Forest']
    sc = 'P2_3%'
    s = S[(S.scenario == sc) & (S.ranking == 'score')]
    appr_all = D[D.scenario == sc].approve_all.mean() / 1000
    for ax, col, ylab in ((axes[0], 'profit_per_1000_mean', 'profit per applicant ($000)'),
                          (axes[1], 'bad_rate_approved_mean', 'bad rate among approved'),
                          (axes[2], 'gain_over_random_mean', 'gain over random selection ($000)')):
        piv = s.pivot(index='k', columns='model', values=col)
        sc_ = 1000 if col != 'bad_rate_approved_mean' else 1
        ax.fill_between(KS, piv.min(axis=1) / sc_, piv.max(axis=1) / sc_, color=MUTED, alpha=0.18, lw=0,
                        label='range of all 10 models')
        for i, m in enumerate(show):
            ax.plot(KS, piv[m] / sc_, marker='o', ms=3.5, color=SERIES[i], label=m)
        ax.set_xlabel('acceptance rate k'); ax.set_ylabel(ylab)
    axes[0].plot(KS, [k * appr_all for k in KS], color=INK, ls='--', lw=1, label='random selection')
    axes[0].axhline(appr_all, color=MUTED, ls=':', lw=1)
    axes[1].axhline(yv.mean(), color=INK, ls='--', lw=1)
    axes[2].axhline(0, color=INK, ls='--', lw=1)
    axes[0].legend(fontsize=7, loc='upper left')
    axes[0].set_title('Profit vs acceptance (D3, P2 3%)'); axes[1].set_title('Bad rate vs acceptance'); axes[2].set_title('Gain over random')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "capacity_d3.png"); plt.close(fig)


if __name__ == "__main__":
    main()
