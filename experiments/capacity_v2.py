"""Phase 3C / task B: dollar capacity and ROI (H12, H13). D3 x10 repeats + OOT folds.
Rankings: default-risk, expected profit, expected ROI (per model) + interest-rate-only
baselines (lowest first, highest first). Random baseline = k x approve-all."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sst
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import RESULTS_DIR, FIGURES_DIR
from creditrisk.calibration import make_calibrators, apply_cal
from creditrisk.capacity import approve_count, approve_dollar, outcome
from creditrisk.data import full_frame, profit_p1, profit_p2
from creditrisk.expected_profit import ExpectedProfit
from creditrisk.models import MODEL_NAMES
from creditrisk.repeats import get_dataset, load_score
from creditrisk.plotting import style, SERIES, MUTED, INK
import matplotlib.pyplot as plt
import oot_scores as OS
from oot_study import fold_indices, usable_columns, FOLDS

KS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SCEN = ['P1', 'P2_3%']


def eval_block(meta, keys, y, P, funded):
    rows = []
    for sc in SCEN:
        p = P[sc]; base = p.sum() / len(p) * 1000; roi_all = p.sum() / funded.sum()
        for (rk, model), key in keys.items():
            kk = key[sc] if isinstance(key, dict) else key
            for cap in ('count', 'dollar'):
                for k in KS:
                    a = approve_count(kk, k) if cap == 'count' else approve_dollar(kk, funded, k)
                    o = outcome(a, y, p, funded)
                    rows.append(dict(**meta, scenario=sc, capacity=cap, ranking=rk, model=model, k=k,
                                     **o, gain_profit=o['profit_per_1000'] - k * base,
                                     gain_roi=o['roi'] - roi_all, approve_all=base, roi_all=roi_all))
    return rows


def make_keys(models_scores, X_te, funded_te, mu):
    keys = {}
    ir = X_te['interest_rate'].values
    for model, (s_te, p_te) in models_scores.items():
        keys[('default_risk', model)] = s_te
        E = {sc: (1 - p_te) * mu[sc][0] + p_te * mu[sc][1] for sc in SCEN}
        keys[('expected_profit', model)] = {sc: -E[sc] for sc in SCEN}
        keys[('expected_roi', model)] = {sc: -E[sc] / funded_te for sc in SCEN}
    keys[('ir_low_first', 'interest_rate')] = ir
    keys[('ir_high_first', 'interest_rate')] = -ir
    return keys


def main():
    t0 = time.time()
    # ------------------------------------------------------------ D3
    X, y, _ = get_dataset('D3'); yv = y.values
    F = full_frame().loc[X.index]; funded = F.funded_amnt.values.astype(float)
    PR = {'P1': profit_p1(F).values, 'P2_3%': profit_p2(F, 0.03).values}
    rows = []; saved = None
    for rep in range(10):
        z0 = load_score('D3', rep, MODEL_NAMES[0], 'none'); tv, te = z0['idx_tv'], z0['idx_te']
        mu = {}
        for sc in SCEN:
            ep = ExpectedProfit().fit(X.iloc[tv], yv[tv], PR[sc][tv], seed=42 + rep, n_jobs=4)
            mu[sc] = ep.mu(X.iloc[te])
        ms = {}
        for m in MODEL_NAMES:
            z = load_score('D3', rep, m, 'none')
            cal = make_calibrators(z['oof'].astype(float), yv[tv])['isotonic']
            s = z['te'].astype(float); ms[m] = (s, apply_cal(cal, s))
        keys = make_keys(ms, X.iloc[te], funded[te], mu)
        rows += eval_block(dict(dataset='D3', fold=f'rep{rep}', rep=rep), keys, yv[te],
                           {sc: PR[sc][te] for sc in SCEN}, funded[te])
        if rep == 0:
            saved = dict(te=te, keys=keys)
        print(f"[cap2] D3 rep {rep} {time.time()-t0:.0f}s", flush=True)
    D = pd.DataFrame(rows)
    # ------------------------------------------------------------ OOT (10 models, seed 42)
    t = OS.oot_table(); Xo, yo, yr, tm = t['X'], t['y'], t['yr'], t['tm']
    orows = []
    for T in FOLDS:
        tr, ca, te = fold_indices(T, yr, tm)
        cols, _ = usable_columns(Xo, tr); Xc = Xo[cols]
        mu = {}
        for sc in SCEN:
            ep = ExpectedProfit().fit(Xc.iloc[tr], yo[tr], t['P'][sc][tr], seed=42, n_jobs=4)
            mu[sc] = ep.mu(Xc.iloc[te])
        ms = {}
        for m in MODEL_NAMES:
            z = OS.load(T, m, 42)
            cal = make_calibrators(z['s_ca'].astype(float), yo[ca])['isotonic']
            s = z['s_te'].astype(float); ms[m] = (s, apply_cal(cal, s))
        keys = make_keys(ms, Xo.iloc[te], t['funded'][te], mu)
        orows += eval_block(dict(dataset='OOT', fold=f'T{T}', rep=T), keys, yo[te],
                            {sc: t['P'][sc][te] for sc in SCEN}, t['funded'][te])
        print(f"[cap2] OOT T={T} {time.time()-t0:.0f}s", flush=True)
    O = pd.DataFrame(orows)
    A = pd.concat([D, O], ignore_index=True)
    keep = ['dataset', 'fold', 'rep', 'scenario', 'capacity', 'ranking', 'model', 'k', 'profit_per_1000', 'roi',
            'bad_count', 'bad_dollar', 'accept_dollar_share', 'gain_profit', 'gain_roi', 'approve_all', 'roi_all']
    A[keep].round(5).to_csv(RESULTS_DIR / "phase3c_capacity.csv", index=False)

    pd.set_option('display.width', 250)
    sc, cap, k = 'P2_3%', 'dollar', 0.5
    d = D[(D.scenario == sc) & (D.capacity == cap) & (D.k == k)]
    g = d.groupby(['ranking', 'model']).agg(profit=('profit_per_1000', 'mean'), roi=('roi', 'mean'),
                                              gain_profit=('gain_profit', 'mean'), gain_roi=('gain_roi', 'mean'),
                                              bad_count=('bad_count', 'mean'), bad_dollar=('bad_dollar', 'mean'))
    print(f"\n=== D3, {sc}, dollar capacity, k=50%: approve-all profit {d.approve_all.mean():,.0f}, ROI {d.roi_all.mean():.4f}")
    print(g.groupby(level=0).agg(['mean']).round(4).to_string())
    print(g.loc[['ir_high_first', 'ir_low_first']].round(4).to_string())
    # ---------------- H12
    e = D[(D.scenario == sc) & (D.ranking == 'expected_profit')]
    gc = e[e.capacity == 'count'].groupby('rep').gain_profit.mean(); gd = e[e.capacity == 'dollar'].groupby('rep').gain_profit.mean()
    diff = gd - gc
    m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff)); tq = sst.t.ppf(.975, len(diff) - 1)
    print(f"\nH12: mean gain count {gc.mean():,.0f} vs dollar {gd.mean():,.0f}; diff(dollar-count) = {m:,.0f} "
          f"CI [{m-tq*se:,.0f}, {m+tq*se:,.0f}] -> {'SUPPORTED' if (m < 0 and m + tq*se < 0) else 'REJECTED'}")
    # ---------------- H13(a)
    dr = D[(D.scenario == sc) & (D.capacity == 'dollar') & (D.k == .5) & (D.ranking == 'default_risk')]
    roi_gain = dr.groupby('model').gain_roi.mean()
    print("H13(a): default-risk ROI gain vs approve-all by model (k=50% dollar):", roi_gain.round(4).to_dict(),
          "-> ", "SUPPORTED" if (roi_gain < 0).all() else "REJECTED", f"({int((roi_gain<0).sum())}/10 negative)")
    # ---------------- H13(b): paired bootstrap on rep 0
    cand = d.groupby(['ranking', 'model']).roi.mean()
    cand = cand[cand.index.get_level_values(0).isin(['default_risk', 'expected_profit', 'expected_roi'])]
    best = cand.idxmax(); print("best model-based ranking by ROI:", best, round(cand.max(), 4))
    te, keys = saved['te'], saved['keys']
    p_te, f_te = PR['P2_3%'][te], funded[te]
    a_ir = approve_dollar(keys[('ir_high_first', 'interest_rate')], f_te, .5)
    kb = keys[best]; kb = kb['P2_3%'] if isinstance(kb, dict) else kb
    a_b = approve_dollar(kb, f_te, .5)
    rng = np.random.default_rng(0); n = len(te); diffs = []
    for _ in range(20):
        W = rng.poisson(1.0, (50, n)).astype(float)
        r_ir = (W @ (p_te * a_ir)) / (W @ (f_te * a_ir)); r_b = (W @ (p_te * a_b)) / (W @ (f_te * a_b))
        diffs.extend(r_ir - r_b)
    diffs = np.array(diffs); roi_all0 = p_te.sum() / f_te.sum()
    lo, hi = np.quantile(diffs, [.025, .975]); marg = 0.02 * roi_all0
    print(f"H13(b): ROI(ir_high) - ROI(best) = {np.mean(diffs):+.4f} CI [{lo:+.4f}, {hi:+.4f}], margin ±{marg:.4f} "
          f"-> {'within margin (equivalent)' if (lo > -marg and hi < marg) else 'NOT within margin'}")
    print(f"   ROI(ir_high)={(p_te[a_ir].sum()/f_te[a_ir].sum()):.4f}  ROI(best)={(p_te[a_b].sum()/f_te[a_b].sum()):.4f}  ROI(all)={roi_all0:.4f}")
    # ---------------- ranking stability
    for rk in ('default_risk', 'expected_profit', 'expected_roi'):
        piv = D[(D.scenario == sc) & (D.capacity == 'dollar') & (D.ranking == rk) & (D.model != 'interest_rate')] \
            .groupby(['model', 'k']).roi.mean().unstack()
        rs = [spearmanr(piv[a], piv[b])[0] for i, a in enumerate(KS) for b in KS[i + 1:]]
        o_ = O[(O.scenario == sc) & (O.capacity == 'dollar') & (O.ranking == rk)].pivot_table(index='model', columns=['fold', 'k'], values='roi')
        fo = sorted(set(c[0] for c in o_.columns))
        r2 = [spearmanr(o_[(f1, kk)], o_[(f2, kk)])[0] for kk in KS for i, f1 in enumerate(fo) for f2 in fo[i + 1:]]
        print(f"ranking={rk}: model-rank Spearman across k {np.mean(rs):.2f}; across OOT folds {np.mean(r2):.2f}")
    print("\nOOT, P2-3%, dollar k=50%: ROI by ranking (mean over models) per fold; ir_high; approve-all")
    o = O[(O.scenario == sc) & (O.capacity == 'dollar') & (O.k == .5)]
    print(o.groupby(['ranking', 'fold']).roi.mean().unstack().round(4).to_string())
    print(o.groupby('fold').roi_all.mean().round(4).to_dict())
    # ---------------- figure
    style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    dd = D[(D.scenario == sc) & (D.capacity == 'dollar')]
    spec = [('default_risk', 'LightGBM', 'default-risk (LightGBM)', SERIES[0]),
            ('expected_profit', 'LightGBM', 'expected profit (LightGBM)', SERIES[1]),
            ('expected_roi', 'LightGBM', 'expected ROI (LightGBM)', SERIES[2]),
            ('ir_high_first', 'interest_rate', 'interest rate, high first', SERIES[3]),
            ('ir_low_first', 'interest_rate', 'interest rate, low first', SERIES[4])]
    base = dd.approve_all.mean() / 1000; roi_a = dd.roi_all.mean()
    for rk, m, lab, col in spec:
        s = dd[(dd.ranking == rk) & (dd.model == m)].groupby('k')[['profit_per_1000', 'roi', 'bad_dollar']].mean()
        axes[0].plot(KS, s.profit_per_1000 / 1000, marker='o', ms=3.5, color=col, label=lab)
        axes[1].plot(KS, s.roi * 100, marker='o', ms=3.5, color=col, label=lab)
        axes[2].plot(KS, s.bad_dollar * 100, marker='o', ms=3.5, color=col, label=lab)
    axes[0].plot(KS, [k_ * base for k_ in KS], color=INK, ls='--', lw=1, label='random selection')
    axes[1].axhline(roi_a * 100, color=INK, ls='--', lw=1)
    axes[0].set_ylabel('profit per applicant ($000)'); axes[1].set_ylabel('ROI (% of funded $)')
    axes[2].set_ylabel('dollar-weighted bad rate (%)')
    for a_, t_ in zip(axes, ('Profit vs dollar capacity', 'ROI vs dollar capacity (dashed = random)', 'Bad rate (dollar-weighted)')):
        a_.set_xlabel('share of funded dollars approved'); a_.set_title(t_)
    axes[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "capacity_v2.png"); plt.close(fig)


if __name__ == "__main__":
    main()
